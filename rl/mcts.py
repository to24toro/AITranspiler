import math
import random
from typing import Callable, Dict, List, Optional

import numpy as np
import tensorflow as tf

from rl.game import Game, encode_state


class MCTS:
    def __init__(self, qubits: int, network: tf.keras.Model, config: Dict):
        """
        Monte Carlo Tree Search (MCTS) implementation.

        Args:
            qubits (int): The number of qubits (dimension of the state matrix).
            network (tf.keras.Model): Neural network for policy and value prediction.
            config (Dict): Configuration dictionary containing MCTS settings.
        """
        self.qubits: int = qubits
        self.network: tf.keras.Model = network
        mcts_settings = config["mcts_settings"]
        self.alpha: float = mcts_settings["dirichlet_alpha"]
        self.c_puct: float = mcts_settings["c_puct"]
        self.epsilon: float = mcts_settings["epsilon"]
        self.max_depth: int = mcts_settings["max_depth"]

        # MCTS state tracking
        self.P: Dict[str, List[float]] = {}  # Prior probabilities
        self.N: Dict[str, List[int]] = {}  # Visit counts
        self.W: Dict[str, List[float]] = {}  # Total action-values
        self.next_states: Dict[str, List[Optional[np.ndarray]]] = {}  # Cached next states

        # State encoding
        self.state_to_str: Callable[[np.ndarray], str] = lambda state: "".join(
            map(str, state.astype(int).flatten().tolist())
        )

        # Game instance
        self.game = Game(qubits, config)

        # Temperature schedule
        self.initial_tau: float = 1.0
        self.final_tau: float = 0.1
        self.tau_decay_steps: int = 100
        self.current_episode: int = 0

    def update_temperature(self) -> float:
        """
        Update the temperature parameter (τ) for exploration.

        Returns:
            float: Updated temperature value.
        """
        decay_factor = min(1, self.current_episode / self.tau_decay_steps)
        return self.initial_tau * (1 - decay_factor) + self.final_tau * decay_factor

    def search(
        self, root_state: np.ndarray, num_simulations: int, prev_action: Optional[int]
    ) -> np.ndarray:
        """
        Perform MCTS simulations from the given root state.

        Args:
            root_state (np.ndarray): The root state to start simulations from.
            num_simulations (int): The number of simulations to run.
            prev_action (Optional[int]): The previous action taken.

        Returns:
            np.ndarray: Normalized visit counts as a policy vector.
        """
        tau = self.update_temperature()
        s = self.state_to_str(root_state)

        if s not in self.P:
            _ = self._expand(root_state, prev_action)

        valid_actions = self.game.get_valid_actions(root_state, prev_action)

        if self.alpha is not None:
            dirichlet_noise = np.random.dirichlet([self.alpha] * len(valid_actions))
            for a, noise in zip(valid_actions, dirichlet_noise):
                self.P[s][a] = (1 - self.epsilon) * self.P[s][a] + self.epsilon * noise

        for _ in range(num_simulations):
            U = [
                self.c_puct
                * self.P[s][a]
                * math.sqrt(sum(self.N[s]))
                / (1 + self.N[s][a])
                for a in range(len(self.game.coupling_map))
            ]
            Q = [
                self.W[s][a] / self.N[s][a] if self.N[s][a] != 0 else 0
                for a in range(len(self.game.coupling_map))
            ]

            scores = np.array(
                [
                    u + q if action in valid_actions else -np.inf
                    for action, (u, q) in enumerate(zip(U, Q))
                ]
            )

            action = random.choice(np.where(scores == np.max(scores))[0])

            next_state = self.next_states[s][action]
            v = self._evaluate(next_state, prev_action=action, max_depth=self.max_depth)

            self.W[s][action] += v
            self.N[s][action] += 1

        visits = np.array([self.N[s][a] for a in range(len(self.game.coupling_map))])
        mcts_policy = np.power(visits, 1 / tau)
        mcts_policy /= np.sum(mcts_policy)

        return mcts_policy

    def _expand(self, state: np.ndarray, prev_action: Optional[int]) -> float:
        """
        Expand a new node by predicting prior probabilities and value.

        Args:
            state (np.ndarray): The state to expand.
            prev_action (Optional[int]): The previous action taken.

        Returns:
            float: Predicted value from the neural network.
        """
        s = self.state_to_str(state)

        with tf.device("/cpu:0"):
            nn_policy, nn_value = self.network.predict(encode_state(state, self.qubits))

        nn_policy = nn_policy[0]
        nn_value = nn_value[0][0]

        self.P[s] = nn_policy
        self.N[s] = [0] * len(self.game.coupling_map)
        self.W[s] = [0] * len(self.game.coupling_map)

        valid_actions = self.game.get_valid_actions(state, prev_action)

        self.next_states[s] = [
            (
                self.game.step(state, action, prev_action)[0]
                if (action in valid_actions)
                else None
            )
            for action in range(len(self.game.coupling_map))
        ]
        return nn_value

    def _evaluate(
        self,
        state: Optional[np.ndarray],
        prev_action: Optional[int] = None,
        total_score: int = 0,
        depth: int = 0,
        max_depth: int = 1000,
    ) -> float:
        """
        Recursively evaluate a node in the search tree.

        Args:
            state (Optional[np.ndarray]): The current state to evaluate.
            prev_action (Optional[int]): The previous action taken.
            total_score (int): The total score accumulated so far.
            depth (int): The current depth of the search.
            max_depth (int): Maximum depth to search.

        Returns:
            float: Predicted value or reward for the node.
        """
        if depth >= max_depth or state is None:
            return -np.inf

        s = self.state_to_str(state)

        if self.game.is_done(state):
            reward = self.game.get_reward(state, total_score)
            return reward

        elif s not in self.P:
            nn_value = self._expand(state, prev_action)
            return nn_value

        else:
            U = [
                self.c_puct
                * self.P[s][a]
                * math.sqrt(sum(self.N[s]))
                / (1 + self.N[s][a])
                for a in range(len(self.game.coupling_map))
            ]
            Q = [
                self.W[s][a] / self.N[s][a] if self.N[s][a] != 0 else 0
                for a in range(len(self.game.coupling_map))
            ]

            valid_actions = self.game.get_valid_actions(state, prev_action)

            scores = np.array(
                [
                    u + q if action in valid_actions else -np.inf
                    for action, (u, q) in enumerate(zip(U, Q))
                ]
            )

            action = random.choice(np.where(scores == np.max(scores))[0])

            next_state = self.next_states[s][action]
            v = self._evaluate(
                next_state,
                prev_action=action,
                total_score=total_score,
                depth=depth + 1,
                max_depth=max_depth,
            )

            self.W[s][action] += v
            self.N[s][action] += 1

            return v