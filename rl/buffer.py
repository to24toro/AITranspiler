import collections
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class Sample:
    state: np.ndarray
    mcts_policy: np.ndarray
    reward: float


class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def get_minibatch(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        states = np.stack(
            [torch.tensor(s.state, dtype=torch.float32)
                .unsqueeze(0) for s in samples], axis=0
        )

        mcts_policy = np.array([s.mcts_policy for s in samples], dtype=np.float32)
        rewards = np.array([s.reward for s in samples], dtype=np.float32).reshape(-1, 1)

        return states, mcts_policy, rewards

    def add_record(self, record: List[Sample]):
        self.buffer.extend(record)
