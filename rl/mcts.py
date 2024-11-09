import math
import random

import numpy as np
import tensorflow as tf
import yaml

from rl import game

# Load MCTS settings from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

mcts_settings = config["mcts_settings"]


class MCTS:
    def __init__(self, network):
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

    def search(self, root_state, num_simulations):
        s = self.state_to_str(root_state)

        if s not in self.P:
            _ = self._expand(root_state)

        valid_actions = list(range(game.ACTION_SPACE))

        #: Adding Dirichlet noise to the prior probabilities in the root node
        if self.alpha is not None:
            dirichlet_noise = np.random.dirichlet([self.alpha] * len(valid_actions))
            for a, noise in zip(valid_actions, dirichlet_noise):
                self.P[s][a] = (1 - self.epsilon) * self.P[s][a] + self.epsilon * noise

        #: MCTS simulation
        for _ in range(num_simulations):
            U = [
                self.c_puct
                * self.P[s][a]
                * math.sqrt(sum(self.N[s]))
                / (1 + self.N[s][a])
                for a in range(game.ACTION_SPACE)
            ]
            Q = [
                self.W[s][a] / self.N[s][a] if self.N[s][a] != 0 else 0
                for a in range(game.ACTION_SPACE)
            ]

            assert len(U) == len(Q) == game.ACTION_SPACE

            scores = [u + q for u, q in zip(U, Q)]

            #: All actions are valid in this game
            action = random.choice(np.where(scores == np.max(scores))[0])

            next_state, _ = self.next_states[s][action]
            v = self._evaluate(next_state, max_depth=self.max_depth)

            self.W[s][action] += v
            self.N[s][action] += 1

        mcts_policy = [self.N[s][a] / sum(self.N[s]) for a in range(game.ACTION_SPACE)]

        return mcts_policy

    def _expand(self, state):
        s = self.state_to_str(state)

        with tf.device("/cpu:0"):
            nn_policy, nn_value = self.network.predict(game.encode_state(state))

        nn_policy = nn_policy.numpy().tolist()[0]
        nn_value = nn_value.numpy()[0][0]

        self.P[s] = nn_policy
        self.N[s] = [0] * game.ACTION_SPACE
        self.W[s] = [0] * game.ACTION_SPACE

        #: Cache next states
        self.next_states[s] = [
            (game.step(state, action)[0:2]) for action in range(game.ACTION_SPACE)
        ]

        return nn_value

    def _evaluate(self, state, depth=0, max_depth=1000):
        if depth >= max_depth:
            return -np.inf

        s = self.state_to_str(state)

        if game.is_done(state):
            reward = game.get_reward(state, 0)
            return reward

        elif s not in self.P:
            nn_value = self._expand(state)
            return nn_value

        else:
            U = [
                self.c_puct
                * self.P[s][a]
                * math.sqrt(sum(self.N[s]))
                / (1 + self.N[s][a])
                for a in range(game.ACTION_SPACE)
            ]
            Q = [
                self.W[s][a] / self.N[s][a] if self.N[s][a] != 0 else 0
                for a in range(game.ACTION_SPACE)
            ]

            scores = [u + q for u, q in zip(U, Q)]
            action = random.choice(np.where(scores == np.max(scores))[0])

            next_state, _ = self.next_states[s][action]

            v = self._evaluate(next_state, depth=depth + 1, max_depth=max_depth)

            self.W[s][action] += v
            self.N[s][action] += 1

            return v
