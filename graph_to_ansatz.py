from typing import List, Tuple

import networkx as nx
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp


class GraphAnsatzConverter:

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.pauli_list = self.build_max_cut_paulis(graph)
        self.hamiltonian = self.cost_hamiltonian(self.pauli_list)

    def build_max_cut_paulis(self, graph: nx.Graph) -> List[Tuple[str, float]]:
        """
        Convert a NetworkX graph to a list of Pauli operators for the Max-Cut problem.

        Parameters:
        - graph: nx.Graph, the input NetworkX graph where nodes are qubits
          and edges represent interactions with weights.

        Returns:
        - pauli_list: List[Tuple[str, float]], list of Pauli operators and corresponding weights.
        """
        pauli_list = []
        for edge in graph.edges(data=True):
            paulis = ["I"] * graph.number_of_nodes()
            paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

            weight = edge[2].get("weight", 1.0)

            pauli_list.append(("".join(paulis)[::-1], weight))

        return pauli_list

    def cost_hamiltonian(self, pauli_list: List[Tuple[str, float]]) -> SparsePauliOp:
        """
        Create a cost Hamiltonian for the Max-Cut problem.

        Parameters:
        - pauli_list: List[Tuple[str, float]], list of Pauli operators and corresponding weights.

        Returns:
        - SparsePauliOp: the cost Hamiltonian.
        """
        return SparsePauliOp.from_list(pauli_list)

    def build_qaoa_ansatz(self, reps: int) -> QAOAAnsatz:
        """
        Build a QAOA ansatz circuit for the Max-Cut problem.

        Parameters:
        - reps: int, the number of repetitions of the QAOA layers.

        Returns:
        - QAOAAnsatz: the QAOA ansatz circuit.
        """
        circuit = QAOAAnsatz(cost_operator=self.hamiltonian, reps=reps)
        circuit.measure_all()
        return circuit
