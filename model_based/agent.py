import math
import os
import torch
import torch.optim as optim
from model_based.model import ValueNetwork, EnvironmentModel
from model_based.replay_buffer import ReplayBuffer
import numpy as np
import torch.nn.functional as F


class MBAgent:
    def __init__(self, name, obs_dim, act_dim, n_agents, gamma=0.95, eps_start=0.99, eps_end=0.05, eps_decay=1000, tau=0.001):
        self.name = name
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.value_network = ValueNetwork(obs_dim).to(self.device)
        # self.value_target = ValueNetwork(obs_dim).to(self.device)
        # self.value_target.load_state_dict(self.value_network.state_dict())

        self.environment_model = EnvironmentModel(obs_dim, act_dim)

        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.0001)
        self.environment_optimizer = optim.Adam(self.environment_model.parameters(), lr=0.0001)

        self.replay = ReplayBuffer()
        self.steps_done = 0

    def act(self, obs, explore=True):
        with torch.no_grad():
            obs_tensor = torch.tensor(np.array(obs), device=self.device)
            next_states, rewards = self.environment_model(obs_tensor)
            next_states = next_states.view(self.act_dim, self.obs_dim)
            next_states_values = self.value_network(next_states).flatten()
            if explore:
                eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
                self.steps_done += 1
                if np.random.random_sample() < eps:
                    return np.random.randint(0, self.act_dim)

            expected_returns = self.gamma * next_states_values + rewards
            return torch.argmax(expected_returns).item()

    def update(self, samples):
        states, actions, rewards, next_states = samples

        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float32)
        actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.int64)
        rewards = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float32)
        # ======== Environment Model update ========

        pred_next_states, pred_rewards = self.environment_model(states)
        # reshape pred_next_states: (B, act_dim * obs_dim) → (B, act_dim, obs_dim)
        pred_next_states = pred_next_states.view(-1, self.act_dim, self.obs_dim)
        # gather predictions for actual actions
        batch_idx = torch.arange(pred_next_states.size(0), device=self.device)
        pred_next_states = pred_next_states[batch_idx, actions]
        pred_rewards = pred_rewards[batch_idx, actions]
        # compute loss and optimize environment model
        next_states_loss = F.mse_loss(pred_next_states, next_states)
        rewards_loss = F.mse_loss(pred_rewards, rewards)
        environment_loss = next_states_loss + rewards_loss
        self.environment_optimizer.zero_grad()
        environment_loss.backward()
        self.environment_optimizer.step()

    def update_value(self, obs, reward, next_state):
        obs = torch.tensor(np.array(obs), device=self.device, dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), device=self.device, dtype=torch.float32)
        # ======== Value Network Update ========
        state_value = self.value_network(obs)
        with torch.no_grad():
            next_state_value = self.value_network(next_state)
            target_state_value = self.gamma * next_state_value + reward

        states_values_loss = F.mse_loss(state_value, target_state_value)
        self.value_optimizer.zero_grad()
        states_values_loss.backward()
        self.value_optimizer.step()
        # target network soft update
    #     self._soft_update(self.value_target, self.value_network)

    # def _soft_update(self, target, source):
    #     for t, s in zip(target.parameters(), source.parameters()):
    #         t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def __repr__(self):
        return self.name + "[obs: " + str(self.obs_dim) + " act: " + str(self.act_dim) + "]"

    def save_model(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.value_network.state_dict(), dir_path+self.name+"_value.pth")
        torch.save(self.environment_model.state_dict(), dir_path+self.name+"_environment.pth")

    def load_model(self, dir_path):
        self.value_network.load_state_dict(torch.load(dir_path+self.name+"_value.pth"))
        self.environment_model.load_state_dict(torch.load(dir_path+self.name+"_environment.pth"))
        self.value_target.load_state_dict(self.value_network.state_dict())
