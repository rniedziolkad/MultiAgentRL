import random
from collections import deque
import torch


class ReplayBuffer:
    def __init__(self, max_size=10**4):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        states, actions, rewards, next_states, finals, istates = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(finals),
            torch.stack(istates)
        )

    def __len__(self):
        return len(self.buffer)
