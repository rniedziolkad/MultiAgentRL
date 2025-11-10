import torch
from independent_dqn.model import DQN
import torch.optim as optim
import numpy as np
import math
import torch.nn.functional as F
import os
from independent_dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, name, obs_dim, act_dim, gamma=0.95, tau=0.001, eps_start=0.99, eps_end=0.05, eps_decay=1000):
        self.name = name
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = DQN(obs_dim, act_dim).to(self.device)
        self.network_target = DQN(obs_dim, act_dim).to(self.device)
        self.network_target.load_state_dict(self.network.state_dict())

        self.network_optimizer = optim.Adam(self.network.parameters(), lr=0.0001)
        self.replay = ReplayBuffer()

    def act(self, obs, explore=True):
        with torch.no_grad():
            obs_tensor = torch.tensor(np.array(obs), device=self.device)
            q_values = self.network(obs_tensor)

            if explore:
                eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
                self.steps_done += 1
                if np.random.random_sample() < eps:
                    return np.random.randint(0, q_values.shape[-1])

            return torch.argmax(q_values).item()

    def update(self, samples):
        states, actions, rewards, next_states = samples

        # ======== DQN update ========
        # compute current qvalues
        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float32)
        actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.int64).unsqueeze(1)

        q_values = self.network(states).gather(dim=1, index=actions)

        # compute target q values
        next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            max_next_q_values = self.network_target(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * max_next_q_values

        # compute loss and optimize model
        loss = F.mse_loss(q_values, target_q_values)
        self.network_optimizer.zero_grad()
        loss.backward()
        self.network_optimizer.step()

        # soft update target network
        self._soft_update(self.network_target, self.network)

    def _soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def __repr__(self):
        return self.name + "[obs: " + str(self.obs_dim) + " act: " + str(self.act_dim) + "]"

    def save_model(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.network.state_dict(), dir_path + self.name + "_network.pth")

    def load_model(self, dir_path):
        self.network.load_state_dict(torch.load(dir_path+self.name+"_network.pth"))
        self.network.load_state_dict(self.network.state_dict())

