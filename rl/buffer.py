import collections
import numpy as np
import game

class ReplayBuffer:
    def __init__(self, buffer_size):
        """Initialize the replay buffer."""
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

    def get_minibatch(self, batch_size):
        """Sample a minibatch of experiences from the buffer."""
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        # Encode states for neural network input
        states = np.stack([game.encode_state(s.state) for s in samples], axis=0)

        mcts_policy = np.array([s.mcts_policy for s in samples], dtype=np.float32)
        rewards = np.array([s.reward for s in samples], dtype=np.float32).reshape(-1, 1)

        return states, mcts_policy, rewards

    def add_record(self, record):
        """Add a list of samples to the buffer."""
        for sample in record:
            self.buffer.append(sample)