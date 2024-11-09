import unittest
from dataclasses import dataclass

import numpy as np

from rl.buffer import ReplayBuffer


@dataclass
class Sample:
    state: np.ndarray
    mcts_policy: np.ndarray
    reward: float


class TestReplayBuffer(unittest.TestCase):

    def setUp(self):
        self.buffer_size = 10
        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size)

    def test_add_record_and_length(self):
        sample = Sample(state=np.zeros((3, 3)), mcts_policy=np.ones(5) / 5, reward=1.0)
        self.replay_buffer.add_record([sample])
        self.assertEqual(len(self.replay_buffer), 1)

    def test_buffer_overflow(self):
        sample = Sample(state=np.zeros((3, 3)), mcts_policy=np.ones(5) / 5, reward=1.0)
        for _ in range(self.buffer_size + 5):
            self.replay_buffer.add_record([sample])
        self.assertEqual(len(self.replay_buffer), self.buffer_size)

    def test_get_minibatch(self):
        sample = Sample(state=np.zeros((3, 3)), mcts_policy=np.ones(5) / 5, reward=1.0)
        self.replay_buffer.add_record([sample] * self.buffer_size)
        batch_size = 4
        states, mcts_policies, rewards = self.replay_buffer.get_minibatch(batch_size)
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(mcts_policies.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(
            states.shape[1:], sample.state.shape + (1,)
        )  # Assuming encode_state adds channel dimension


if __name__ == "__main__":
    unittest.main()
