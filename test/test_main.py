import unittest

import ray
import yaml

from rl import game
from rl.main import main, selfplay
from rl.network import ResNet


class TestMainFunctions(unittest.TestCase):

    def setUp(self):
        # Load settings
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        game.initialize_game()
        self.network = ResNet(action_space=game.ACTION_SPACE)
        # Initialize network weights
        state = game.get_initial_state()
        dummy_input = game.encode_state(state)
        self.network.predict(dummy_input)
        self.current_weights = self.network.get_weights()

    def test_selfplay(self):
        future = selfplay.remote(self.current_weights, test=True)
        record = ray.get(future)
        self.assertIsInstance(record, list)
        self.assertGreater(len(record), 0)
        # Check if each sample in record is of correct type
        from dataclasses import is_dataclass

        self.assertTrue(is_dataclass(record[0]))
        ray.shutdown()

    def test_main(self):
        # Since main() runs indefinitely, we can test initial setup
        try:
            main(test=True)
            success = True
        except Exception as e:
            success = False
            print(f"main() function raised an exception: {e}")
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
