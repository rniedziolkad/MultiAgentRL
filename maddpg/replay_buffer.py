import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=10**4):
        self.buffer = deque(maxlen=max_size)

    def add(self, obs, actions, rewards, next_obs):
        obs_array = np.stack([obs[name] for name in obs], axis=0)
        next_obs_array = np.stack([next_obs[name] for name in obs], axis=0)

        actions_array = np.array([actions[name] for name in actions], dtype=np.int64)
        rewards_array = np.array([rewards[name] for name in rewards], dtype=np.float32)

        self.buffer.append((obs_array, actions_array, rewards_array, next_obs_array))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = map(np.array, zip(*batch))
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)
