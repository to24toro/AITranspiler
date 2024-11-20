from dataclasses import dataclass
from itertools import combinations
from threading import Timer

import networkx as nx
import numpy as np
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    Commuting2qGateRouter,
    FindCommutingPauliEvolutions,
    SwapStrategy,
)


@dataclass
class SATResult:
    """A data class to hold the result of a SAT solver."""

    satisfiable: (
        bool  # Satisfiable is True if the SAT model could be solved in a given time.
    )
    solution: dict  # The solution to the SAT problem if it is satisfiable.
    mapping: (
        list  # The mapping of nodes in the pattern graph to nodes in the target graph.
    )
    elapsed_time: float  # The time it took to solve the SAT model.


class SATMapper:
    r"""A class to introduce a SAT-approach to solve
    the initial mapping problem in SWAP gate insertion for commuting gates.

    When this pass is run on a DAG it will look for the first instance of
    :class:`.Commuting2qBlock` and use the program graph :math:`P` of this block of gates to
    find a layout for a given swap strategy. This layout is found with a
    binary search over the layers :math:`l` of the swap strategy. At each considered layer
    a subgraph isomorphism problem formulated as a SAT is solved by a SAT solver. Each instance
    is whether it is possible to embed the program graph :math:`P` into the effective
    connectivity graph :math:`C_l` that is achieved by applying :math:`l` layers of the
    swap strategy to the coupling map :math:`C_0` of the backend. Since solving SAT problems
    can be hard, a ``time_out`` fixes the maximum time allotted to the SAT solver for each
    instance. If this time is exceeded the considered problem is deemed unsatisfiable and
    the binary search proceeds to the next number of swap layers :math:``l``.
    """

    def __init__(self, timeout: int = 60):
        """Initialize the SATMapping.

        Args:
            timeout: The allowed time in seconds for each iteration of the SAT solver. This
                variable defaults to 60 seconds.
        """
        self.timeout = timeout

    def find_initial_mappings(
        self,
        program_graph: nx.Graph,
        swap_strategy: SwapStrategy,
        min_layers: int | None = None,
        max_layers: int | None = None,
    ) -> dict[int, SATResult]:
        r"""Find an initial mapping for a given swap strategy. Perform a binary search
        over the number of swap layers, and for each number of swap layers solve a
        subgraph isomorphism problem formulated as a SAT problem.

        Args:
            program_graph (nx.Graph): The program graph with commuting gates, where
                                        each edge represents a two-qubit gate.
            swap_strategy (SwapStrategy): The swap strategy to use to find the initial mapping.
            min_layers (int): The minimum number of swap layers to consider. Defaults to
            the maximum degree of the program graph - 2.
            max_layers (int): The maximum number of swap layers to consider. Defaults to
            the number of qubits in the swap strategy - 2.

        Returns:
            dict[int, SATResult]: A dictionary containing the results of the SAT solver for
                                    each number of swap layers.
        """
        num_nodes_g1 = len(program_graph.nodes())
        num_nodes_g2 = swap_strategy.distance_matrix.shape[0]
        if num_nodes_g1 > num_nodes_g2:
            return SATResult(False, [], [], 0)
        if min_layers is None:
            # use the maximum degree of the program graph - 2 as the lower bound.
            min_layers = max((d for _, d in program_graph.degree)) - 2
        if max_layers is None:
            max_layers = num_nodes_g2 - 1

        variable_pool = IDPool(start_from=1)
        variables = np.array(
            [
                [variable_pool.id(f"v_{i}_{j}") for j in range(num_nodes_g2)]
                for i in range(num_nodes_g1)
            ],
            dtype=int,
        )
        vid2mapping = {v: idx for idx, v in np.ndenumerate(variables)}
        binary_search_results = {}

        def interrupt(solver):
            # This function is called to interrupt the solver when the timeout is reached.
            solver.interrupt()

        # Make a cnf (conjunctive normal form) for the one-to-one mapping constraint
        cnf1 = []
        for i in range(num_nodes_g1):
            clause = variables[i, :].tolist()
            cnf1.append(clause)
            for k, m in combinations(clause, 2):
                cnf1.append([-1 * k, -1 * m])
        for j in range(num_nodes_g2):
            clause = variables[:, j].tolist()
            for k, m in combinations(clause, 2):
                cnf1.append([-1 * k, -1 * m])

        # Perform a binary search over the number of swap layers to find the minimum
        # number of swap layers that satisfies the subgraph isomorphism problem.
        while min_layers < max_layers:
            num_layers = (min_layers + max_layers) // 2

            # Create the connectivity matrix. Note that if the swap strategy cannot reach
            # full connectivity then its distance matrix will have entries with -1. These
            # entries must be treated as False.
            d_matrix = swap_strategy.distance_matrix
            connectivity_matrix = ((-1 < d_matrix) & (d_matrix <= num_layers)).astype(
                int
            )
            # Make a cnf for the adjacency constraint
            cnf2 = []
            for e_0, e_1 in list(program_graph.edges):
                clause_matrix = np.multiply(connectivity_matrix, variables[e_1, :])
                clause = np.concatenate(
                    (
                        [[-variables[e_0, i]] for i in range(num_nodes_g2)],
                        clause_matrix,
                    ),
                    axis=1,
                )
                # Remove 0s from each clause
                cnf2.extend([c[c != 0].tolist() for c in clause])

            cnf = CNF(from_clauses=cnf1 + cnf2)

            with Solver(bootstrap_with=cnf, use_timer=True) as solver:
                # Solve the SAT problem with a timeout.
                # Timer is used to interrupt the solver when the timeout is reached.
                timer = Timer(self.timeout, interrupt, [solver])
                timer.start()
                status = solver.solve_limited(expect_interrupt=True)
                timer.cancel()
                # Get the solution and the elapsed time.
                sol = solver.get_model()
                e_time = solver.time()

                print(f"Layers: {num_layers}, Status: {status}, Time: {e_time}")
                if status:
                    # If the SAT problem is satisfiable, convert the solution to a mapping.
                    mapping = [vid2mapping[idx] for idx in sol if idx > 0]
                    binary_search_results[num_layers] = SATResult(
                        status, sol, mapping, e_time
                    )
                    max_layers = num_layers
                else:
                    # If the SAT problem is unsatisfiable, return the last satisfiable solution.
                    binary_search_results[num_layers] = SATResult(
                        status, sol, [], e_time
                    )
                    min_layers = num_layers + 1

        return binary_search_results

    def remap_graph_with_sat(self, graph: nx.Graph, swap_strategy, max_layers):
        """Applies the SAT mapping.

        Args:
            graph (nx.Graph): The graph to remap.
            swap_strategy (SwapStrategy): The swap strategy to use to find the initial mapping.

        Returns:
            tuple: A tuple containing the remapped graph, the edge map, and the number of layers of
            the swap strategy that was used to find the initial mapping. If no solution is found
            then the tuple contains None for each element.
            Note the returned edge map `{k: v}` means that node `k` in the original
            graph gets mapped to node `v` in the Pauli strings.
        """
        num_nodes = len(graph.nodes())
        results = self.find_initial_mappings(graph, swap_strategy, 0, max_layers)
        solutions = [k for k, v in results.items() if v.satisfiable]

        if len(solutions):
            min_k = min(solutions)
            edge_map = dict(results[min_k].mapping)
            # Create the remapped graph
            remapped_graph = nx.Graph()
            remapped_graph.add_nodes_from(range(num_nodes))
            mapping = dict(results[min_k].mapping)
            for i, graph_edge in enumerate(list(graph.edges)):
                remapped_edge = tuple(mapping[node] for node in graph_edge)
                remapped_graph.add_edge(*remapped_edge)
            return remapped_graph, edge_map, min_k
        else:
            return None, None, None


def apply_swap_strategy(
    circuit: QuantumCircuit,
    swap_strategy: SwapStrategy,
    edge_coloring: dict[tuple[int, int], int] | None = None,
) -> QuantumCircuit:
    """Transpile with a SWAP strategy.

    Returns:
        A quantum circuit transpiled with the given swap strategy.
    """

    pm_pre = PassManager(
        [
            FindCommutingPauliEvolutions(),
            Commuting2qGateRouter(
                swap_strategy,
                edge_coloring,
            ),
        ]
    )
    return pm_pre.run(circuit)


def edge_coloring(graph):
    line_graph = nx.line_graph(graph)

    coloring = nx.coloring.greedy_color(line_graph)
    edge_coloring = {edge: color for edge, color in coloring.items()}

    return edge_coloring


def apply_qaoa_layers(
    cost_layer: QuantumCircuit,
    meas_map: dict,
    num_layers: int,
    gamma: list[float] | ParameterVector = None,
    beta: list[float] | ParameterVector = None,
    initial_state: QuantumCircuit = None,
    mixer: QuantumCircuit = None,
):
    """Applies QAOA layers to construct circuit.

    First, the initial state is applied. If `initial_state` is None we begin in the
    initial superposition state. Next, we alternate between layers of the cost operator
    and the mixer. The cost operator is alternatively applied in order and in reverse
    instruction order. This allows us to apply the swap strategy on odd `p` layers
    and undo the swap strategy on even `p` layers.
    """

    num_qubits = cost_layer.num_qubits
    new_circuit = QuantumCircuit(num_qubits, num_qubits)

    if initial_state is not None:
        new_circuit.append(initial_state, range(num_qubits))
    else:
        # all h state by default
        new_circuit.h(range(num_qubits))

    if gamma is None or beta is None:
        gamma = ParameterVector("γ", num_layers)
        if mixer is None or mixer.num_parameters == 0:
            beta = ParameterVector("β", num_layers)
        else:
            beta = ParameterVector("β", num_layers * mixer.num_parameters)

    if mixer is not None:
        mixer_layer = mixer
    else:
        mixer_layer = QuantumCircuit(num_qubits)
        mixer_layer.rx(-2 * beta[0], range(num_qubits))

    for layer in range(num_layers):
        bind_dict = {cost_layer.parameters[0]: gamma[layer]}
        cost_layer_ = cost_layer.assign_parameters(bind_dict)
        bind_dict = {
            mixer_layer.parameters[i]: beta[layer + i]
            for i in range(mixer_layer.num_parameters)
        }
        layer_mixer = mixer_layer.assign_parameters(bind_dict)

        if layer % 2 == 0:
            new_circuit.append(cost_layer_, range(num_qubits))
        else:
            new_circuit.append(cost_layer_.reverse_ops(), range(num_qubits))

        new_circuit.append(layer_mixer, range(num_qubits))

    for qidx, cidx in meas_map.items():
        new_circuit.measure(qidx, cidx)

    return new_circuit


def make_meas_map(circuit: QuantumCircuit) -> dict:
    """Return a mapping from qubit index (the key) to classical bit (the value).

    This allows us to account for the swapping order introduced by the SwapStrategy.
    """
    creg = circuit.cregs[0]
    qreg = circuit.qregs[0]

    meas_map = {}
    for inst in circuit.data:
        if inst.operation.name == "measure":
            meas_map[qreg.index(inst.qubits[0])] = creg.index(inst.clbits[0])

    return meas_map


def create_qaoa_swap_circuit(
    cost_operator: SparsePauliOp,
    swap_strategy: SwapStrategy,
    edge_coloring: dict = None,
    theta: list[float] = None,
    qaoa_layers: int = 1,
    initial_state: QuantumCircuit = None,
    mixer: QuantumCircuit = None,
):
    """Create the circuit for QAOA.

    Notes: This circuit construction for QAOA works for quadratic terms in `Z` and will be
    extended to first-order terms in `Z`. Higher-orders are not supported.

    Args:
        cost_operator: the cost operator.
        swap_strategy: selected swap strategy
        edge_coloring: A coloring of edges that should correspond to the coupling
            map of the hardware. It defines the order in which we apply the Rzz
            gates. This allows us to choose an ordering such that `Rzz` gates will
            immediately precede SWAP gates to leverage CNOT cancellation.
        theta: The QAOA angles.
        qaoa_layers: The number of layers of the cost-operator and the mixer operator.
        initial_state: The initial state on which we apply layers of cost-operator
            and mixer.
        mixer: The QAOA mixer. It will be applied as is onto the QAOA circuit. Therefore,
            its output must have the same ordering of qubits as its input.
    """

    num_qubits = cost_operator.num_qubits

    if theta is not None:
        gamma = theta[: len(theta) // 2]
        beta = theta[len(theta) // 2 :]
        qaoa_layers = len(theta) // 2
    else:
        gamma = beta = None

    # First, create the ansatz of 1 layer of QAOA without mixer
    cost_layer = QAOAAnsatz(
        cost_operator,
        reps=1,
        initial_state=QuantumCircuit(num_qubits),
        mixer_operator=QuantumCircuit(num_qubits),
    ).decompose()

    # This will allow us to recover the permutation of the measurements that the swap introduce.
    cost_layer.measure_all()

    # Now, apply the swap strategy for commuting pauli evolution gates
    cost_layer = apply_swap_strategy(cost_layer, swap_strategy, edge_coloring)

    # Compute the measurement map (qubit to classical bit).
    # we will apply this for qaoa_layers % 2 == 1.
    if qaoa_layers % 2 == 1:
        meas_map = make_meas_map(cost_layer)
    else:
        meas_map = {idx: idx for idx in range(num_qubits)}

    cost_layer.remove_final_measurements()

    # Finally, introduce the mixer circuit and add measurements following measurement map
    circuit = apply_qaoa_layers(
        cost_layer, meas_map, qaoa_layers, gamma, beta, initial_state, mixer
    )

    return circuit
