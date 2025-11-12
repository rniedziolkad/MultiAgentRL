import math
import os
import torch
import torch.optim as optim
from model_based.model import ValueNetwork, EnvironmentModel
from model_based.replay_buffer import ReplayBuffer
import numpy as np


class MBAgent:
    def __init__(self, name, obs_dim, act_dim, n_agents, gamma=0.95, eps_start=0.99, eps_end=0.05, eps_decay=1000):
        self.name = name
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.value_network = ValueNetwork(obs_dim).to(self.device)
        self.value_target = ValueNetwork(obs_dim).to(self.device)
        self.value_target.load_state_dict(self.value_network.state_dict())

        self.environment_model = EnvironmentModel(obs_dim, act_dim)

        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.0001)
        self.environment_optimizer = optim.Adam(self.environment_model.parameters(), lr=0.0001)

        self.replay = ReplayBuffer()
        self.steps_done = 0

    def act(self, obs, explore=True):
        with torch.no_grad():
            obs_tensor = torch.tensor(np.arraay(obs), device=self.device)
            next_states, rewards, final = self.environment_model(obs_tensor)
            next_states_values = self.value_network(next_states)
            if explore:
                eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
                self.steps_done += 1
                if np.random.random_sample() < eps:
                    return np.random.randint(0, next_states.shape[-1])

            expected_returns = self.gamma * next_states_values * (1-final) + rewards
            return torch.argmax(expected_returns)

    def update(self, samples):
        pass

    def __repr__(self):
        return self.name + "[obs: " + str(self.obs_dim) + " act: " + str(self.act_dim) + "]"

    def save_model(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.value_network.state_dict(), dir_path+self.name+"_value.pth")
        torch.save(self.environment_model.state_dict(), dir_path+self.name+"_environment.pth")

    def load_model(self, dir_path):
        self.value_network.load_state_dict(torch.load(dir_path+self.name+"_value.pth"))
        self.environment_model.load_state_dict(torch.load(dir_path+self.name+"+_environment.pth"))
        self.value_target.load_state_dict(self.value_network.state_dict())
