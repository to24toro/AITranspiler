import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt  # Added import for plotting

# Game settings
N = 6  # Size of the matrix (N x N upper triangular matrix)
a = 1  # Points for each action (added each time a column pair is selected)
b = 2  # Penalty points when reusing columns included in the set
MAX_STEPS = 50  # Maximum number of steps to forcibly end the game

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
def step(mat, action, coupling_map_mat, used_columns_set):
    """
    Generates a new state based on the current matrix `mat` and the selected action `action`.
    Also calculates the score according to the action.

    Parameters:
    - mat: Current state matrix
    - action: Selected column pair (tuple)
    - coupling_map_mat: Matrix generated from coupling_map
    - used_columns_set: Set of used columns

    Returns:
    - new_mat: Updated matrix
    - action_score: Points gained in this step
    """
    col1, col2 = action
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
    return new_mat, action_score

# Function to check if the game is finished
def is_done(mat):
    """
    Checks whether all elements of the matrix `mat` are zero.

    Returns:
    - True: All elements are zero, game ends
    - False: There are still non-zero elements
    """
    return np.all(mat == 0)

# Function to save the state matrix as an image
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
    # plt.imshow(mat, cmap='gray_r', interpolation='nearest')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"State at step {step_num} saved to {filename}")

# Function to play the game
def play_game():
    """
    Executes the game and simulates the steps until all elements of the matrix become zero.
    """
    mat = get_initial_state()
    coupling_map = get_coupling_map()
    coupling_map_mat = get_coupling_map_mat(coupling_map)
    used_columns_set = set()
    total_score = 0
    actions_taken = []
    step_count = 0  # Step counter

    print("Initial matrix:")
    print(mat)

    # Save initial state as an image
    save_state(mat, step_count)

    while not is_done(mat) and step_count < MAX_STEPS:
        valid_actions = get_valid_actions(coupling_map)
        # Randomly select an action (to be optimized with AlphaZero)
        action = valid_actions[np.random.randint(len(valid_actions))]
        actions_taken.append(action)
        mat, action_score = step(mat, action, coupling_map_mat, used_columns_set)
        total_score += action_score
        step_count += 1

        # Save current state as an image
        save_state(mat, step_count)

        # Debug output
        print(f"\nStep {step_count}")
        print(f"Action taken: {action}, Current score: {total_score}")
        print(f"Current matrix:\n{mat}")

    if is_done(mat):
        print(f"\nGame finished successfully in {step_count} steps with total score: {total_score}")
    else:
        print(f"\nGame terminated after reaching the maximum steps ({MAX_STEPS}).")
        print(f"Current score: {total_score}")

    print(f"Actions taken: {actions_taken}")

if __name__ == "__main__":
    play_game()