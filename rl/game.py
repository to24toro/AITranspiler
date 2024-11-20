import functools
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Load game settings from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

game_settings = config["game_settings"]
N = game_settings["N"]  # Size of the matrix (N x N  matrix)
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


# Initialize the coupling map and related variables
def initialize_game():
    global coupling_map, coupling_map_mat, ACTIONS, ACTION_SPACE
    coupling_map = get_coupling_map()
    coupling_map_mat = get_coupling_map_mat(coupling_map)
    ACTIONS = coupling_map
    # ACTIONS = list(get_valid_actions(state=None,prev_action=None))
    ACTION_SPACE = len(ACTIONS)


def reset_used_columns():
    global used_columns_set
    global used_pair
    used_columns_set = set()
    used_pair = deque()


def _generate_binary_symmetric_matrix():
    upper_triangle = np.triu(np.random.randint(0, 2, size=(N, N)), k=1)
    symmetric_matrix = upper_triangle + upper_triangle.T
    np.fill_diagonal(symmetric_matrix, 0)
    return symmetric_matrix


# Function to initialize an  matrix
def get_initial_state():
    """
    Generates an N x N  matrix consisting of random 0s and 1s.
    """

    mat = _generate_binary_symmetric_matrix()
    mat = mat - np.multiply(mat, coupling_map_mat)
    return mat


def get_initial_test_state():
    """
    Generates an N x N  matrix consisting of random 0s and 1s.
    """

    mat = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ]
    )
    mat = mat - np.multiply(mat, coupling_map_mat)
    return mat


# Function to generate coupling_map
def get_coupling_map():
    """
    Generates a set of possible column pairs.
    Here, we use all adjacent column pairs as an example.
    """
    coupling_map = set()
    for i in range(N - 1):
        coupling_map.add((i, i + 1))
    return list(coupling_map)


# Function to generate a matrix from coupling_map
def get_coupling_map_mat(coupling_map):
    """
    Generates an  matrix from coupling_map.
    For each column pair (i, j) in coupling_map, sets the position (i, j) in the matrix to 1.
    """
    coupling_map_mat = np.zeros((N, N))
    for i, j in coupling_map:
        coupling_map_mat[i, j] = 1
        coupling_map_mat[j, i] = 1
    return coupling_map_mat


# Function to get valid actions
def get_valid_actions(state=None, prev_action=None):
    """
    Retrieves a list of possible actions (column pairs) from coupling_map.
    """
    if state is None:
        return list(range(ACTION_SPACE))
    valid_actions = [
        action
        for action in range(ACTION_SPACE)
        if is_valid_action(state, action, prev_action)
    ]
    if not valid_actions:
        return list(range(ACTION_SPACE))
    return list(valid_actions)


def is_valid_action(state, action, prev_action):
    col1, col2 = coupling_map[action]
    if np.all(state[:, col1] == 0) and np.all(state[:, col2] == 0):
        return False
    if action == prev_action:
        return False
    return True


# Function to update the state
def step(mat, action, prev_action, mcts_policy=None):
    """
    Generates a new state based on the current matrix `mat` and the selected action `action`.
    Also calculates the score according to the action.

    Parameters:
    - mat: Current state matrix
    - action: Selected action index (integer)

    Returns:
    - new_mat: Updated matrix
    - action_score: Points gained in this step
    - done: Boolean indicating if the game is finished
    """
    try:
        assert action in get_valid_actions(mat, prev_action)
    except:
        print(action, prev_action)
        print(mat)
        print(get_valid_actions(mat, prev_action))
        action = np.random.choice(get_valid_actions(mat, prev_action))
    global used_columns_set
    global used_pair
    col1, col2 = ACTIONS[action]
    if col1 > col2:
        col1, col2 = col2, col1
    if is_done(mat):
        return mat, True, 0
    # while (
    #     mcts_policy is not None
    #     and ((np.all(mat[:, col1] == 0) and np.all(mat[:, col2] == 0))
    #     or (used_pair and (col1, col2) == used_pair[-1]))
    # ):
    #     print(mat)
    #     action = np.random.choice(range(ACTION_SPACE), p=mcts_policy)
    #     col1, col2 = ACTIONS[action]
    #     print(mcts_policy)
    #     print(col1,col2)
    #     # print(col1,col2)

    # used_pair.append((col1,col2))
    new_mat = mat.copy()
    # Swap columns
    new_mat[:, [col1, col2]] = new_mat[:, [col2, col1]]
    new_mat[[col1, col2], :] = new_mat[[col2, col1], :]
    # Element-wise multiplication and subtraction
    new_mat = new_mat - np.multiply(new_mat, coupling_map_mat)
    new_mat = np.clip(new_mat, 0, 1)  # Prevent elements from becoming negative
    # Calculate the score
    # action_score = 0
    action_score = a
    if col1 in used_columns_set or col2 in used_columns_set:
        # action_score += b
        used_columns_set.clear()
    used_columns_set.update([col1, col2])
    done = is_done(new_mat)
    return new_mat, done, action_score


# Function to check if the game is finished
def is_done(mat):
    """
    Checks whether all elements of the matrix `mat` are zero.

    Returns:
    - True: All elements are zero, game ends
    - False: There are still non-zero elements
    """
    return np.all(mat == 0)


# Function to get the reward
def get_reward(mat, total_score):
    """
    Calculates the reward for the given state.

    Parameters:
    - mat: Current state matrix
    - total_score: Total score accumulated so far

    Returns:
    - reward: The reward for the current state
    """
    if is_done(mat):
        # If the game is finished, give a high positive reward
        # reward = 100 - total_score  # ゲームクリア時の報酬から合計スコアを引く
        reward = 1 - total_score
    else:
        # Game not finished yet
        reward = -total_score  # 現在の合計スコアの負の値
        # reward = 0
    return reward


# Function to encode the state for neural network input
def encode_state(mat):
    """
    Encodes the state matrix into a format suitable for neural network input.

    Parameters:
    - mat: Current state matrix

    Returns:
    - encoded_state: A numpy array suitable for network input
    """
    # Reshape to (N, N, 1) and normalize if necessary
    encoded_state = mat.reshape(N, N, 1).astype(np.float32)
    return encoded_state


# Function to save the state matrix as an image (optional)
def save_state(mat, step_num):
    """
    Saves the current state matrix `mat` as an image.

    Elements with 0 are white, elements with 1 are black.

    Parameters:
    - mat: Current state matrix
    - step_num: Current step number
    """
    filename = f"state_step_{step_num}.png"
    plt.figure(figsize=(6, 6))
    plt.imshow(mat, cmap="gray_r", interpolation="nearest")
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"State at step {step_num} saved to {filename}")


# Initialize game variables
initialize_game()
