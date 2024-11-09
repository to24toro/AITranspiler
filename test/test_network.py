import unittest

import numpy as np
import tensorflow as tf

from rl import game
from rl.network import ResNet


class TestResNet(unittest.TestCase):

    def setUp(self):
        # Initialize game settings
        game.initialize_game()
        self.action_space = game.ACTION_SPACE
        self.network = ResNet(action_space=self.action_space)
        # Initialize network with dummy input
        state = game.get_initial_state()
        dummy_input = game.encode_state(state)
        self.network.predict(dummy_input)

    def test_network_initialization(self):
        self.assertEqual(self.network.action_space, self.action_space)
        self.assertIsInstance(self.network.res_blocks, list)
        self.assertEqual(len(self.network.res_blocks), self.network.n_blocks)

    def test_network_call(self):
        state = game.get_initial_state()
        input_tensor = game.encode_state(state)
        policy_output, value_output = self.network(np.array([input_tensor]))
        self.assertEqual(policy_output.shape, (1, self.action_space))
        self.assertEqual(value_output.shape, (1, 1))
        # Check if policy outputs probabilities
        self.assertAlmostEqual(np.sum(policy_output.numpy()), 1.0, places=4)
        self.assertTrue(all(0 <= p <= 1 for p in policy_output.numpy().flatten()))
        # Check if value output is in the range [-1, 1]
        self.assertTrue(-1 <= value_output.numpy()[0][0] <= 1)

    def test_network_predict(self):
        state = game.get_initial_state()
        input_tensor = game.encode_state(state)
        policy_output, value_output = self.network.predict(input_tensor)
        self.assertEqual(policy_output.shape, (1, self.action_space))
        self.assertEqual(value_output.shape, (1, 1))

    def test_resblock(self):
        res_block = self.network.res_blocks[0]
        input_tensor = tf.random.uniform((1, game.N, game.N, self.network.filters))
        output_tensor = res_block(input_tensor)
        self.assertEqual(output_tensor.shape, input_tensor.shape)


if __name__ == "__main__":
    unittest.main()
