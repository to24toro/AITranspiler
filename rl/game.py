from collections import deque
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

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
- Add `gate` points for each action (selection of a column pair).
- Save the columns used in the action in the set `used_columns_set` and `used_pair`.
- If a column included in this set is used again in a new action,
  add `layer` penalty points and reset the set to empty.
- The total points accumulated until the end of the game are the final score.
- The goal is to minimize the score.
"""

class Game:
    def __init__(self, qubits: int, config: Dict):
        """
        Initialize the game.

        Args:
            qubits (int): The number of qubits (matrix size N x N).
            config (Dict): Configuration dictionary containing game settings.
        """
        self.gate: int = config["game_settings"]["gate"]
        self.layer: int = config["game_settings"]["layer"]
        self.MAX_STEPS: int = config["game_settings"]["MAX_STEPS"]

        self.qubits: int = qubits
        self.coupling_map: List[Tuple[int, int]] = self._generate_coupling_map()
        self.coupling_map_mat: np.ndarray = self._generate_coupling_map_mat()
        self.state: np.ndarray = self.get_initial_state()
        self.used_pair: deque = deque()
        self.used_columns_set: set = set()

    def _generate_coupling_map(self) -> List[Tuple[int, int]]:
        """
        Generate all adjacent column pairs.

        Returns:
            List[Tuple[int, int]]: A list of tuples representing column pairs.
        """
        return [(i, i + 1) for i in range(self.qubits - 1)]

    def _generate_coupling_map_mat(self) -> np.ndarray:
        """
        Generate a matrix representation of the coupling map.

        Returns:
            np.ndarray: A matrix where each element indicates adjacency (1) or not (0).
        """
        coupling_map_mat = np.zeros((self.qubits, self.qubits))
        for i, j in self.coupling_map:
            coupling_map_mat[i, j] = 1
            coupling_map_mat[j, i] = 1
        return coupling_map_mat

    def reset_used_columns(self) -> None:
        """
        Reset the set of used columns.
        """
        self.used_columns_set.clear()
        self.used_pair.clear()

    def get_initial_state(self) -> np.ndarray:
        """
        Generate an initial random symmetric matrix.

        Returns:
            np.ndarray: A symmetric matrix initialized with 0s and 1s.
        """
        upper_triangle = np.triu(
            np.random.randint(0, 2, size=(self.qubits, self.qubits)), k=1
        )
        symmetric_matrix = upper_triangle + upper_triangle.T
        np.fill_diagonal(symmetric_matrix, 0)
        mat = symmetric_matrix - np.multiply(symmetric_matrix, self.coupling_map_mat)
        return mat

    def get_initial_test_state(self) -> np.ndarray:
        """
        Generate a fixed test matrix.

        Returns:
            np.ndarray: A predefined 4x4 test matrix.
        """
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

    def get_valid_actions(
        self, state: Optional[np.ndarray] = None, prev_action: Optional[int] = None
    ) -> List[int]:
        """
        Retrieve valid actions based on the current state.

        Args:
            state (Optional[np.ndarray]): Current state matrix. Defaults to None.
            prev_action (Optional[int]): Previously taken action. Defaults to None.

        Returns:
            List[int]: A list of valid action indices.
        """
        if state is None:
            return list(range(len(self.coupling_map)))

        valid_actions = [
            action
            for action in range(len(self.coupling_map))
            if self.is_valid_action(state, action, prev_action)
        ]
        return valid_actions if valid_actions else list(range(len(self.coupling_map)))

    def is_valid_action(
        self, state: np.ndarray, action: int, prev_action: Optional[int]
    ) -> bool:
        """
        Check if an action is valid.

        Args:
            state (np.ndarray): Current state matrix.
            action (int): Action to validate.
            prev_action (Optional[int]): Previous action.

        Returns:
            bool: True if action is valid, False otherwise.
        """
        col1, col2 = self.coupling_map[action]
        if np.all(state[:, col1] == 0) and np.all(state[:, col2] == 0):
            return False
        if action == prev_action:
            return False
        return True

    def step(
        self, mat: np.ndarray, action: int, prev_action: Optional[int]
    ) -> Tuple[np.ndarray, bool, int]:
        """
        Perform an action and update the state.

        Args:
            mat (np.ndarray): Current state matrix.
            action (int): Selected action.
            prev_action (Optional[int]): Previous action.

        Returns:
            Tuple[np.ndarray, bool, int]:
                - Updated state matrix.
                - Boolean indicating if the game is done.
                - Score for the action.
        """
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

        action_score = self.gate
        if col1 in self.used_columns_set or col2 in self.used_columns_set:
            action_score += self.layer
            self.reset_used_columns()
        self.used_columns_set.update([col1, col2])

        done = self.is_done(new_mat)
        return new_mat, done, action_score

    def is_done(self, mat: np.ndarray) -> bool:
        """
        Check whether all elements of the matrix are zero.

        Args:
            mat (np.ndarray): State matrix.

        Returns:
            bool: True if all elements are zero, False otherwise.
        """
        return np.all(mat == 0)

    def get_reward(self, mat: np.ndarray, total_score: int) -> int:
        """
        Calculate the reward for the current state.

        Args:
            mat (np.ndarray): Current state matrix.
            total_score (int): Total accumulated score.

        Returns:
            int: Calculated reward.
        """
        return 1 - total_score if self.is_done(mat) else -total_score

    def save_state(self, mat: np.ndarray, step_num: int) -> None:
        """
        Save the current state matrix as an image.

        Args:
            mat (np.ndarray): Current state matrix.
            step_num (int): Current step number.
        """
        filename = f"state_step_{step_num}.png"
        plt.figure(figsize=(6, 6))
        plt.imshow(mat, cmap="gray_r", interpolation="nearest")
        plt.axis("off")
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"State at step {step_num} saved to {filename}")


def encode_state(mat: np.ndarray, qubits: int) -> np.ndarray:
    """
    Encode the state matrix for neural network input.

    Args:
        mat (np.ndarray): State matrix.
        qubits (int): Number of qubits.

    Returns:
        np.ndarray: Encoded state with shape (qubits, qubits, 1).
    """
    return mat.reshape(qubits, qubits, 1).astype(np.float32)