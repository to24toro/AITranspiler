import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt
import yaml

# Load game settings from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

game_settings = config['game_settings']
N = game_settings['N']  # Size of the matrix (N x N upper triangular matrix)
a = game_settings['a']  # Points for each action (added each time a column pair is selected)
b = game_settings['b']  # Penalty points when reusing columns included in the set
MAX_STEPS = game_settings['MAX_STEPS']  # Maximum number of steps to forcibly end the game

# Define global variables
coupling_map = None
coupling_map_mat = None
used_columns_set = set()
"""
Game Description:
The objective of this game is to turn all elements of an N x N upper triangular matrix `mat` consisting of 0s and 1s into 0.

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
- Save the columns used in the action in the set `used_columns_set`.
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
    ACTIONS = list(get_valid_actions(coupling_map))
    ACTION_SPACE = len(ACTIONS)

def reset_used_columns_set():
    global used_columns_set
    used_columns_set = set()

# Function to initialize an upper triangular matrix
def get_initial_state():
    """
    Generates an N x N upper triangular matrix consisting of random 0s and 1s.
    """
    mat = np.triu(np.random.randint(0, 2, size=(N, N)))
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
    return coupling_map

# Function to generate a matrix from coupling_map
def get_coupling_map_mat(coupling_map):
    """
    Generates an upper triangular matrix from coupling_map.
    For each column pair (i, j) in coupling_map, sets the position (i, j) in the matrix to 1.
    """
    coupling_map_mat = np.zeros((N, N))
    for (i, j) in coupling_map:
        coupling_map_mat[i, j] = 1
    return np.triu(coupling_map_mat)  # Make it an upper triangular matrix

# Function to get valid actions
def get_valid_actions(coupling_map):
    """
    Retrieves a list of possible actions (column pairs) from coupling_map.
    """
    return list(coupling_map)

# Function to update the state
def step(mat, action):
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
    global used_columns_set
    col1, col2 = ACTIONS[action]
    new_mat = mat.copy()
    # Swap columns
    new_mat[:, [col1, col2]] = new_mat[:, [col2, col1]]
    # Element-wise multiplication and subtraction
    new_mat = new_mat - np.multiply(new_mat, coupling_map_mat)
    new_mat = np.clip(new_mat, 0, 1)  # Prevent elements from becoming negative
    # Calculate the score
    action_score = a
    if col1 in used_columns_set or col2 in used_columns_set:
        action_score += b
        used_columns_set.clear()
    used_columns_set.update([col1, col2])
    done = is_done(new_mat)
    return new_mat, action_score, done

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
        reward = 100 - total_score
    else:
        # Otherwise, negative reward proportional to the total score
        reward = -total_score
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
    plt.imshow(mat, cmap='gray_r', interpolation='nearest')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"State at step {step_num} saved to {filename}")

# Initialize game variables
initialize_game()