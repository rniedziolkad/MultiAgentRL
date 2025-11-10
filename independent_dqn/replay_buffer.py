import random
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size=10**4):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*samples)
        return (
            states,
            actions,
            rewards,
            next_states
        )

    def __len__(self):
        return len(self.buffer)
