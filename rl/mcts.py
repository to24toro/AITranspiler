import math
import random
from typing import Callable, Dict, List, Optional

import numpy as np
import tensorflow as tf

from rl.game import Game, encode_state


class MCTS:
    def __init__(self, qubits: int, network: tf.keras.Model, config: Dict):
        self.qubits: int = qubits
        self.network: tf.keras.Model = network
        mcts_settings = config["mcts_settings"]
        self.alpha: float = mcts_settings["dirichlet_alpha"]
        self.c_puct: float = mcts_settings["c_puct"]
        self.epsilon: float = mcts_settings["epsilon"]
        self.max_depth: int = mcts_settings["max_depth"]
        self.num_mcts_simulations: int = mcts_settings["num_mcts_simulations"]
        self.tau_threshold: int = mcts_settings.get("tau_threshold", 10)

        self.P: Dict[str, np.ndarray] = {}
        self.N: Dict[str, np.ndarray] = {}
        self.W: Dict[str, np.ndarray] = {}
        self.Q: Dict[str, np.ndarray] = {}
        self.V: Dict[str, float] = {}

        self.state_to_str: Callable[[np.ndarray], str] = lambda state: state.tobytes()

        self.game = Game(qubits, config)

    def search(self, root_state: np.ndarray, prev_action: Optional[int], step_count: int) -> np.ndarray:
        s = self.state_to_str(root_state)

        for _ in range(self.num_mcts_simulations):
            self._simulate(root_state, prev_action)

        N_s = self.N[s]
        if step_count < self.tau_threshold:
            tau = 1.0
        else:
            tau = 0.1

        N_s_tau = N_s ** (1 / tau)
        policy = N_s_tau / np.sum(N_s_tau)
        return policy

    def _simulate(self, state: np.ndarray, prev_action: Optional[int]):
        visited_states = []
        s = self.state_to_str(state)
        done = False
        total_score = 0
        depth = 0
        while True:
            if s not in self.P:
                value = self._expand(state, prev_action)
                self._backup(visited_states, value)
                return
            valid_actions = self.game.get_valid_actions(state, prev_action)
            U = self.c_puct * self.P[s] * np.sqrt(np.sum(self.N[s])) / (1 + self.N[s])
            Q = self.Q[s]
            scores = Q + U
            scores = np.where([a in valid_actions for a in range(len(self.game.coupling_map))], scores, -np.inf)
            action = np.argmax(scores)
            visited_states.append((s, action))
            next_state, done, action_score = self.game.step(state, action, prev_action)
            total_score += action_score
            if done or depth >= self.max_depth:
                reward = self.game.get_reward(next_state, total_score)
                self._backup(visited_states, reward)
                return
            state = next_state
            prev_action = action
            s = self.state_to_str(state)
            depth += 1

    def _expand(self, state: np.ndarray, prev_action: Optional[int]) -> float:
        s = self.state_to_str(state)
        nn_policy, nn_value = self.network.predict(encode_state(state, self.qubits))
        nn_policy = nn_policy[0]
        nn_value = nn_value[0][0]

        valid_actions = self.game.get_valid_actions(state, prev_action)
        policy = np.zeros(len(self.game.coupling_map))
        policy[valid_actions] = nn_policy[valid_actions]
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy /= policy_sum
        else:
            policy[valid_actions] = 1 / len(valid_actions)

        if self.epsilon > 0:
            dirichlet_noise = np.random.dirichlet([self.alpha] * len(valid_actions))
            policy[valid_actions] = (1 - self.epsilon) * policy[valid_actions] + self.epsilon * dirichlet_noise

        self.P[s] = policy
        self.N[s] = np.zeros(len(self.game.coupling_map))
        self.W[s] = np.zeros(len(self.game.coupling_map))
        self.Q[s] = np.zeros(len(self.game.coupling_map))
        self.V[s] = nn_value
        return nn_value

    def _backup(self, visited_states: List[tuple], value: float):
        for s, a in reversed(visited_states):
            self.N[s][a] += 1
            self.W[s][a] += value
            self.Q[s][a] = self.W[s][a] / self.N[s][a]