import collections
from typing import List, Tuple
import numpy as np

from rl.game import encode_state


class ReplayBuffer:
    def __init__(self, buffer_size: int):
        """
        Initialize the replay buffer.
        
        Args:
            buffer_size (int): The maximum size of the buffer.
        """
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self) -> int:
        """
        Return the current size of the buffer.
        
        Returns:
            int: The number of elements currently stored in the buffer.
        """
        return len(self.buffer)

    def get_minibatch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a minibatch of experiences from the buffer.

        Args:
            batch_size (int): The number of samples to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - states (np.ndarray): A batch of encoded states, shape (batch_size, state_dim, ...).
                - mcts_policy (np.ndarray): A batch of MCTS policies, shape (batch_size, action_dim).
                - rewards (np.ndarray): A batch of rewards, shape (batch_size, 1).
        """
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        # Encode states for neural network input
        states = np.stack(
            [encode_state(s.state, s.state.shape[0]) for s in samples], axis=0
        )

        mcts_policy = np.array([s.mcts_policy for s in samples], dtype=np.float32)
        rewards = np.array([s.reward for s in samples], dtype=np.float32).reshape(-1, 1)

        return states, mcts_policy, rewards

    def add_record(self, record: List):
        """
        Add a list of samples to the buffer.

        Args:
            record (List): A list of experience samples to add to the buffer.
        """
        for sample in record:
            self.buffer.append(sample)