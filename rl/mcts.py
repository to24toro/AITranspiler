import math
import random

import numpy as np
import tensorflow as tf
import yaml

from rl.game import Game, encode_state

# Load MCTS settings from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

mcts_settings = config["mcts_settings"]


class MCTS:
    def __init__(self, qubits, network):
        self.qubits = qubits
        self.network = network
        self.alpha = mcts_settings["dirichlet_alpha"]
        self.c_puct = mcts_settings["c_puct"]
        self.epsilon = mcts_settings["epsilon"]
        self.max_depth = mcts_settings["max_depth"]

        #: Prior probability
        self.P = {}

        #: Visit count
        self.N = {}

        #: W is total action-value and Q is mean action-value
        self.W = {}

        #: Cache next states to save computation
        self.next_states = {}

        #: Use JSON string of the state matrix as the key
        self.state_to_str = lambda state: "".join(
            map(str, state.astype(int).flatten().tolist())
        )

        self.game = Game(qubits, config)
        self.initial_tau = 1.0  # 初期の温度パラメータ
        self.final_tau = 0.1  # 最終的な温度パラメータ
        self.tau_decay_steps = 100  # 温度パラメータを減衰させるエピソード数
        self.current_episode = 0

    def update_temperature(self):
        decay_factor = min(1, self.current_episode / self.tau_decay_steps)
        return self.initial_tau * (1 - decay_factor) + self.final_tau * decay_factor

    def search(self, root_state, num_simulations, prev_action):
        tau = self.update_temperature()
        s = self.state_to_str(root_state)

        if s not in self.P:
            _ = self._expand(root_state, prev_action)

        valid_actions = self.game.get_valid_actions(root_state, prev_action)

        if self.alpha is not None:
            dirichlet_noise = np.random.dirichlet([self.alpha] * len(valid_actions))
            for a, noise in zip(valid_actions, dirichlet_noise):
                self.P[s][a] = (1 - self.epsilon) * self.P[s][a] + self.epsilon * noise

        # MCTSシミュレーション
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

            assert len(U) == len(Q) == len(self.game.coupling_map)

            scores = [u + q for u, q in zip(U, Q)]

            # 無効なアクションのスコアをマイナス無限大に設定
            scores = np.array(
                [
                    score if action in valid_actions else -np.inf
                    for action, score in enumerate(scores)
                ]
            )

            action = random.choice(np.where(scores == np.max(scores))[0])

            next_state = self.next_states[s][action]
            v = self._evaluate(next_state, prev_action=action, max_depth=self.max_depth)

            self.W[s][action] += v
            self.N[s][action] += 1

        # MCTSポリシーを温度パラメータで調整
        visits = np.array([self.N[s][a] for a in range(len(self.game.coupling_map))])
        mcts_policy = np.power(visits, 1 / tau)
        mcts_policy /= np.sum(mcts_policy)  # 正規化して確率分布に

        return mcts_policy

    def _expand(self, state, prev_action):
        s = self.state_to_str(state)

        with tf.device("/cpu:0"):
            nn_policy, nn_value = self.network.predict(encode_state(state, self.qubits))

        nn_policy = nn_policy.numpy().tolist()[0]
        nn_value = nn_value.numpy()[0][0]

        self.P[s] = nn_policy
        self.N[s] = [0] * len(self.game.coupling_map)
        self.W[s] = [0] * len(self.game.coupling_map)

        valid_actions = self.game.get_valid_actions(state, prev_action)
        #: Cache next states
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
        self, state, prev_action=None, total_score=0, depth=0, max_depth=1000
    ):
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
            assert len(U) == len(Q) == len(self.game.coupling_map)

            valid_actions = self.game.get_valid_actions(state, prev_action)

            scores = [u + q for u, q in zip(U, Q)]
            scores = np.array(
                [
                    score if action in valid_actions else -np.inf
                    for action, score in enumerate(scores)
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
