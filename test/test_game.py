import unittest

import numpy as np

from rl import game


class TestGameFunctions(unittest.TestCase):

    def setUp(self):
        # Initialize game settings
        game.initialize_game()
        game.reset_used_columns()

    def test_get_initial_state(self):
        state = game.get_initial_state()
        self.assertEqual(state.shape, (game.N, game.N))
        # Check if elements are 0 or 1
        self.assertTrue(np.all(np.isin(state, [0, 1])))

    def test_get_coupling_map(self):
        coupling_map = game.get_coupling_map()
        self.assertIsInstance(coupling_map, set)
        self.assertGreater(len(coupling_map), 0)
        # Check if coupling_map contains tuples of integers
        for pair in coupling_map:
            self.assertIsInstance(pair, tuple)
            self.assertEqual(len(pair), 2)
            self.assertIsInstance(pair[0], int)
            self.assertIsInstance(pair[1], int)

    def test_get_coupling_map_mat(self):
        coupling_map = game.get_coupling_map()
        coupling_map_mat = game.get_coupling_map_mat(coupling_map)
        self.assertEqual(coupling_map_mat.shape, (game.N, game.N))
        # Check if elements are 0 or 1
        self.assertTrue(np.all(np.isin(coupling_map_mat, [0, 1])))

    def test_get_valid_actions(self):
        valid_actions = game.get_valid_actions(game.coupling_map)
        self.assertIsInstance(valid_actions, list)
        self.assertGreater(len(valid_actions), 0)
        # Check if valid_actions matches ACTIONS
        self.assertEqual(valid_actions, game.ACTIONS)

    def test_step(self):
        state = game.get_initial_state()
        action = 0  # Assuming at least one action exists
        new_state, action_score, done = game.step(state, action)
        self.assertEqual(new_state.shape, (game.N, game.N))
        self.assertIsInstance(action_score, int)
        self.assertIsInstance(done, np.bool_)
        # Check if new_state elements are 0 or 1
        self.assertTrue(np.all(np.isin(new_state, [0, 1])))

    def test_is_done(self):
        zero_state = np.zeros((game.N, game.N))
        self.assertTrue(game.is_done(zero_state))
        non_zero_state = zero_state.copy()
        non_zero_state[0, 1] = 1
        self.assertFalse(game.is_done(non_zero_state))

    def test_get_reward(self):
        zero_state = np.zeros((game.N, game.N))
        total_score = 10
        reward = game.get_reward(zero_state, total_score)
        self.assertEqual(reward, 100)
        non_zero_state = zero_state.copy()
        non_zero_state[0, 1] = 1
        reward = game.get_reward(non_zero_state, total_score)
        self.assertEqual(reward, -total_score)

    def test_encode_state(self):
        state = game.get_initial_state()
        encoded_state = game.encode_state(state)
        self.assertEqual(encoded_state.shape, (game.N, game.N, 1))
        self.assertEqual(encoded_state.dtype, np.float32)

    def test_reset_used_columns(self):
        game.used_columns_set.add(1)
        game.used_pair.append([1,2])
        game.reset_used_columns()
        self.assertEqual(len(game.used_columns_set), 0)
        self.assertEqual(len(game.used_pair), 0)

    def test_save_state(self):
        # This test will check if the function runs without error
        state = game.get_initial_state()
        try:
            game.save_state(state, step_num=0)
            success = True
        except Exception as e:
            success = False
            print(f"save_state function raised an exception: {e}")
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
