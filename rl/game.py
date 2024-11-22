import functools
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Load game settings from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

game_settings = config["game_settings"]
a = game_settings[
    "a"
]  # Points for each action (added each time a column pair is selected)
b = game_settings["b"]  # Penalty points when reusing columns included in the set
MAX_STEPS = game_settings[
    "MAX_STEPS"
]  # Maximum number of steps to forcibly end the game

# Define global variables
coupling_map = None
coupling_map_mat = None
used_columns_set = set()
used_pair = deque()
"""
Game Description:
The objective of this game is to turn all elements of an N x N  matrix `mat` consisting of 0s and 1s into 0.

Game Procedure:
1. Select a pair of columns (i, j) from the `coupling_map`.
2. Swap columns i and j of the matrix `mat`.
3. Let the swapped matrix be `mat'`, and using the matrix `coupling_map_mat` generated from `coupling_map`,
   perform element-wise multiplication and subtraction.
   The calculation is: `mat' = mat' - mat' * coupling_map_mat`
4. Update the current state `mat` with the resulting matrix `mat'`.
5. Repeat steps 2 to 4 until all elements of the matrix `mat` become 0 to clear the game.

Scoring:
- Add `a` points for each action (selection of a column pair).
- Save the columns used in the action in the set `used_columns_set` and `used_pair`.
- If a column included in this set is used again in a new action,
  add `b` penalty points and reset the set to empty.
- The total points accumulated until the end of the game are the final score.
- The goal is to minimize the score.
"""


class Game:
    def __init__(self, qubits, config):
        # Load game settings
        self.a = config["game_settings"]["a"]
        self.b = config["game_settings"]["b"]
        self.MAX_STEPS = config["game_settings"]["MAX_STEPS"]

        # Initialize variables
        self.qubits = qubits
        self.coupling_map = self._generate_coupling_map()
        self.coupling_map_mat = self._generate_coupling_map_mat()
        self.state = self.get_initial_state()
        self.used_pair = deque()
        self.used_columns_set = set()

    def _generate_coupling_map(self):
        """Generate all adjacent column pairs."""
        return [(i, i + 1) for i in range(self.qubits - 1)]

    def _generate_coupling_map_mat(self):
        """Generate matrix representation of the coupling map."""
        coupling_map_mat = np.zeros((self.qubits, self.qubits))
        for i, j in self.coupling_map:
            coupling_map_mat[i, j] = 1
            coupling_map_mat[j, i] = 1
        return coupling_map_mat

    def reset_used_columns(self):
        """Reset the set of used columns."""
        self.used_columns_set.clear()
        self.used_pair.clear()

    def get_initial_state(self):
        """Generate an initial random symmetric matrix."""
        upper_triangle = np.triu(
            np.random.randint(0, 2, size=(self.qubits, self.qubits)), k=1
        )
        symmetric_matrix = upper_triangle + upper_triangle.T
        np.fill_diagonal(symmetric_matrix, 0)
        mat = symmetric_matrix - np.multiply(symmetric_matrix, self.coupling_map_mat)
        return mat

    def get_initial_test_state(self):
        """Generate a fixed test matrix."""
        mat = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 0],
            ]
        )
        mat = mat - np.multiply(mat, self.coupling_map_mat)
        return mat

    def get_valid_actions(self, state=None, prev_action=None):
        """Retrieve valid actions based on the current state."""
        if state is None:
            return list(range(len(self.coupling_map)))

        valid_actions = [
            action
            for action in range(len(self.coupling_map))
            if self.is_valid_action(state, action, prev_action)
        ]
        return valid_actions if valid_actions else list(range(len(self.coupling_map)))

    def is_valid_action(self, state, action, prev_action):
        """Check if an action is valid."""
        col1, col2 = self.coupling_map[action]
        if np.all(state[:, col1] == 0) and np.all(state[:, col2] == 0):
            return False
        if action == prev_action:
            return False
        return True

    def step(self, mat, action, prev_action):
        """Perform an action and update the state."""
        if action not in self.get_valid_actions(mat, prev_action):
            action = np.random.choice(self.get_valid_actions(mat, prev_action))

        col1, col2 = self.coupling_map[action]
        if col1 > col2:
            col1, col2 = col2, col1

        if self.is_done(mat):
            return mat, True, 0

        new_mat = mat.copy()
        new_mat[:, [col1, col2]] = new_mat[:, [col2, col1]]
        new_mat[[col1, col2], :] = new_mat[[col2, col1], :]
        new_mat = new_mat - np.multiply(new_mat, self.coupling_map_mat)
        new_mat = np.clip(new_mat, 0, 1)

        action_score = self.a
        if col1 in self.used_columns_set or col2 in self.used_columns_set:
            action_score += self.b
            self.reset_used_columns()
        self.used_columns_set.update([col1, col2])

        done = self.is_done(new_mat)
        return new_mat, done, action_score

    def is_done(self, mat):
        """
        Checks whether all elements of the matrix `mat` are zero.

        Returns:
        - True: All elements are zero, game ends
        - False: There are still non-zero elements
        """
        return np.all(mat == 0)

    def get_reward(self, mat, total_score):
        """Calculate the reward for the current state."""
        return 1 - total_score if self.is_done(mat) else -total_score

    def save_state(self, mat, step_num):
        """Save the current state matrix as an image."""
        filename = f"state_step_{step_num}.png"
        plt.figure(figsize=(6, 6))
        plt.imshow(mat, cmap="gray_r", interpolation="nearest")
        plt.axis("off")
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"State at step {step_num} saved to {filename}")


def encode_state(mat, qubits):
    """Encode the state matrix for neural network input."""
    return mat.reshape(qubits, qubits, 1).astype(np.float32)
