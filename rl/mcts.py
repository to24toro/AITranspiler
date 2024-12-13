import math
import random
from typing import Callable, Dict

import numpy as np
import tf_keras as keras

from rl.game import Game, encode_state

class MCTS:
    def __init__(self, qubits: int, network: keras.Model, config: Dict):
        """
        Initialize the Monte Carlo Tree Search (MCTS) instance.

        :param qubits: Number of qubits in the game.
        :param network: Neural network model for policy and value prediction.
        :param config: Configuration dictionary containing MCTS and game settings.
        """
        self.qubits: int = qubits
        self.network: keras.Model = network
        mcts_settings = config["mcts_settings"]
        self.alpha: float = mcts_settings["dirichlet_alpha"]  # Dirichlet noise parameter.
        self.c_puct: float = mcts_settings["c_puct"]  # Exploration/exploitation parameter.
        self.epsilon: float = mcts_settings["epsilon"]  # Weight for Dirichlet noise.
        self.max_depth: int = mcts_settings["max_depth"]  # Maximum search depth.
        self.num_mcts_simulations: int = mcts_settings["num_mcts_simulations"]  # Simulations per search.
        self.tau_threshold: int = mcts_settings.get("tau_threshold", 10)  # Tau threshold for temperature decay.

        self.next_states = {}  # Stores next states for each state-action pair (lazily evaluated).
        self.P: Dict[str, np.ndarray] = {}  # Policy probabilities for each state.
        self.N: Dict[str, np.ndarray] = {}  # Visit counts for each state-action pair.
        self.W: Dict[str, np.ndarray] = {}  # Total value for each state-action pair.
        self.Q: Dict[str, np.ndarray] = {}  # Mean value (Q-value) for each state-action pair.
        self.V: Dict[str, float] = {}  # Value estimates for each state.

        # Converts a state matrix to a unique string representation.
        self.state_to_str: Callable[[np.ndarray], str] = lambda state: "".join(
            map(str, state.astype(int).flatten().tolist())
        )

        self.game = Game(qubits, config)

        # Temperature parameters for controlling exploration.
        self.initial_tau = 1.0
        self.final_tau = 0.1
        self.tau_decay_steps = 100
        self.current_episode = 0

    def update_temperature(self):
        """
        Update the temperature value for exploration based on the current episode.
        """
        decay_factor = min(1, self.current_episode / self.tau_decay_steps)
        return self.initial_tau * (1 - decay_factor) + self.final_tau * decay_factor

    def search(self, root_state, num_simulations, prev_action):
        """
        Perform MCTS simulations starting from the root state.

        :param root_state: The initial state matrix for the search.
        :param num_simulations: Number of MCTS simulations to perform.
        :param prev_action: Previous action taken (if any).
        :return: A policy distribution over actions based on visit counts.
        """
        tau = self.update_temperature()
        s = self.state_to_str(root_state)

        if s not in self.P:
            _ = self._expand(root_state, prev_action)

        valid_actions = self.game.get_valid_actions(root_state, prev_action)

        if self.alpha is not None and len(valid_actions) > 0:
            # Add Dirichlet noise to the policy for exploration.
            dirichlet_noise = np.random.dirichlet([self.alpha] * len(valid_actions))
            for a, noise in zip(valid_actions, dirichlet_noise):
                self.P[s][a] = (1 - self.epsilon) * self.P[s][a] + self.epsilon * noise
            self.P[s] /= np.sum(self.P[s])

        for _ in range(num_simulations):
            # Calculate U and Q values for all actions.
            U = [
                self.c_puct
                * self.P[s][a]
                * math.sqrt(sum(self.N[s]) + 1e-8)
                / (1 + self.N[s][a])
                for a in range(self.game.action_space)
            ]
            Q = [
                self.W[s][a] / self.N[s][a] if self.N[s][a] != 0 else 0
                for a in range(self.game.action_space)
            ]

            scores = [u + q for u, q in zip(U, Q)]

            # Mask invalid actions by setting their scores to negative infinity.
            scores = np.array(
                [
                    score if action in valid_actions else -np.inf
                    for action, score in enumerate(scores)
                ]
            )

            # Select the best action based on the scores.
            action_candidates = np.where(scores == np.max(scores))[0]
            action = random.choice(action_candidates)

            # Lazy evaluation: If next_state not computed yet, compute it now.
            if self.next_states[s][action] is None:
                next_state, done, _ = self.game.step(root_state, action, prev_action)
                self.next_states[s][action] = next_state
            else:
                next_state = self.next_states[s][action]

            v = self._evaluate(next_state, prev_action=action, max_depth=self.max_depth)

            # Update W and N values for the selected action.
            self.W[s][action] += v
            self.N[s][action] += 1

        # Compute the policy distribution based on visit counts.
        visits = np.array([self.N[s][a] for a in range(self.game.action_space)])
        mcts_policy = np.power(visits, 1 / tau)
        mcts_policy /= np.sum(mcts_policy)

        return mcts_policy

    def _expand(self, state, prev_action):
        """
        Expand a state node in the tree by initializing its attributes.

        :param state: The state matrix to expand.
        :param prev_action: Previous action taken (if any).
        :return: The neural network's value prediction for the state.
        """
        s = self.state_to_str(state)

        nn_policy, nn_value = self.network.predict(encode_state(state, self.qubits))

        nn_policy = nn_policy.numpy()[0]
        nn_value = nn_value.numpy()[0][0]

        self.P[s] = nn_policy
        self.N[s] = np.zeros(self.game.action_space, dtype=np.float32)
        self.W[s] = np.zeros(self.game.action_space, dtype=np.float32)

        # Initialize next_states lazily. None means not computed yet.
        self.next_states[s] = [None] * self.game.action_space

        return nn_value

    def _evaluate(self, state, prev_action=None, total_score=0, depth=0, max_depth=1000):
        """
        Evaluate a state by recursively simulating games up to a maximum depth.

        :param state: Current state matrix.
        :param prev_action: Previous action taken (if any).
        :param total_score: Accumulated score so far.
        :param depth: Current recursion depth.
        :param max_depth: Maximum recursion depth allowed.
        :return: The estimated value of the state.
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
                * math.sqrt(sum(self.N[s]) + 1e-8)
                / (1 + self.N[s][a])
                for a in range(self.game.action_space)
            ]
            Q = [
                self.W[s][a] / self.N[s][a] if self.N[s][a] != 0 else 0
                for a in range(self.game.action_space)
            ]

            scores = [u + q for u, q in zip(U, Q)]
            valid_actions = self.game.get_valid_actions(state, prev_action)

            scores = np.array(
                [
                    score if action in valid_actions else -np.inf
                    for action, score in enumerate(scores)
                ]
            )

            # Select the best action
            best_actions = np.where(scores == np.max(scores))[0]
            action = random.choice(best_actions)

            # Lazy evaluation of next_state
            if self.next_states[s][action] is None:
                next_state, done, score = self.game.step(state, action, prev_action)
                self.next_states[s][action] = next_state
            else:
                next_state = self.next_states[s][action]

            v = self._evaluate(
                next_state,
                prev_action=action,
                total_score=total_score,
                depth=depth + 1,
                max_depth=max_depth,
            )

            # Update W and N
            self.W[s][action] += v
            self.N[s][action] += 1

            return v