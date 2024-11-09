import json
import unittest

import numpy as np

from rl import game
from rl.mcts import MCTS
from rl.network import ResNet


class TestMCTS(unittest.TestCase):

    def setUp(self):
        # Initialize game and network
        game.initialize_game()
        game.reset_used_columns_set()
        self.state = game.get_initial_state()
        self.network = ResNet(action_space=game.ACTION_SPACE)
        # Initialize network weights
        dummy_input = game.encode_state(self.state)
        self.network.predict(dummy_input)

    def test_mcts_init(self):
        mcts = MCTS(network=self.network)
        self.assertIsNotNone(mcts.network)
        self.assertIsInstance(mcts.P, dict)
        self.assertIsInstance(mcts.N, dict)
        self.assertIsInstance(mcts.W, dict)
        self.assertIsInstance(mcts.next_states, dict)

    def test_mcts_search(self):
        mcts = MCTS(network=self.network)
        num_simulations = 10
        mcts_policy = mcts.search(
            root_state=self.state, num_simulations=num_simulations
        )
        self.assertEqual(len(mcts_policy), game.ACTION_SPACE)
        self.assertAlmostEqual(sum(mcts_policy), 1.0, places=4)
        self.assertTrue(all(0 <= p <= 1 for p in mcts_policy))

    def test_mcts_expand(self):
        mcts = MCTS(network=self.network)
        nn_value = mcts._expand(self.state)
        s = mcts.state_to_str(self.state)
        self.assertIn(s, mcts.P)
        self.assertIn(s, mcts.N)
        self.assertIn(s, mcts.W)
        self.assertIn(s, mcts.next_states)
        self.assertIsInstance(nn_value, (float, np.float64, np.float32))

    def test_mcts_evaluate(self):
        mcts = MCTS(network=self.network)
        value = mcts._evaluate(self.state)
        self.assertIsInstance(value, (float, np.float64, np.float32))

    def test_state_to_str(self):
        mcts = MCTS(network=self.network)
        s = mcts.state_to_str(self.state)
        self.assertIsInstance(s, str)
        # Convert back to array and compare
        state_list = json.loads(s)
        state_array = np.array(state_list)
        self.assertTrue(np.array_equal(state_array, self.state))


if __name__ == "__main__":
    unittest.main()
