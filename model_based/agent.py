import math
import os
import torch
import torch.optim as optim
from model_based.model import ValueNetwork, EnvironmentModel, InternalStateDecoder
from model_based.replay_buffer import ReplayBuffer
import numpy as np
import torch.nn.functional as F


class MBAgent:
    def __init__(self, name, obs_dim, act_dim, gamma=0.95, eps_start=0.99, eps_end=0.05, eps_decay=1000, tau=0.002):
        self.name = name
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.internal_state_dim = 128

        self.value_network = ValueNetwork(self.internal_state_dim).to(self.device)
        self.value_target = ValueNetwork(self.internal_state_dim).to(self.device)
        self.value_target.load_state_dict(self.value_network.state_dict())

        self.environment_model = EnvironmentModel(obs_dim, self.internal_state_dim, act_dim).to(self.device)
        self.internal_state_decoder = InternalStateDecoder(self.internal_state_dim, obs_dim).to(self.device)

        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.0001)
        self.environment_optimizer = optim.Adam(self.environment_model.parameters(), lr=0.0001)
        self.decoder_optimizer = optim.Adam(self.internal_state_decoder.parameters(), lr=0.0001)

        self.replay = ReplayBuffer()
        self.steps_done = 0

    def act(self, obs, internal_state, explore=True):
        with torch.inference_mode():
            obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            next_istates = self.environment_model(obs_tensor, internal_state)
            next_istates = next_istates.view(self.act_dim, self.internal_state_dim)

            next_states_values = self.value_network(next_istates).flatten()
            _, rewards, finals = self.internal_state_decoder(next_istates)
            rewards = rewards.flatten()
            finals = torch.sigmoid(finals.flatten())
            if explore:
                eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
                self.steps_done += 1
                if np.random.random_sample() < eps:
                    action = np.random.randint(0, self.act_dim)
                    return action, next_istates[action]

            expected_returns = self.gamma * next_states_values * (1.0 - finals) + rewards
            action = torch.argmax(expected_returns).item()
            return action, next_istates[action]

    def update(self, samples):
        observations, actions, rewards, next_observations, finals, istates = samples
        # ======== Environment Model update ========

        next_istates = self.environment_model(observations, istates)
        # reshape pred_next_states: (B, act_dim * obs_dim) → (B, act_dim, obs_dim)
        next_istates = next_istates.view(-1, self.act_dim, self.internal_state_dim)
        # gather predictions for actual actions
        batch_idx = torch.arange(next_istates.size(0), device=self.device)
        next_istates = next_istates[batch_idx, actions]
        pred_next_obs, pred_rewards, pred_finals = self.internal_state_decoder(next_istates)
        # compute loss and optimize environment model
        next_states_loss = F.mse_loss(pred_next_obs, next_observations)
        rewards_loss = F.mse_loss(pred_rewards.flatten(), rewards)
        # pos_weight = torch.tensor([250.0], device=self.device)
        finals_loss = F.binary_cross_entropy_with_logits(pred_finals.flatten(), finals)
        environment_loss = next_states_loss + rewards_loss + finals_loss
        self.environment_optimizer.zero_grad()
        self.internal_state_decoder.zero_grad()
        environment_loss.backward()
        self.environment_optimizer.step()
        self.decoder_optimizer.step()

    def update_value(self, istate, reward, next_istate, final):
        # ======== Value Network Update ========
        state_value = self.value_network(istate)
        with torch.no_grad():
            next_state_value = self.value_target(next_istate)
            target_state_value = self.gamma * next_state_value * (1.0 - final) + reward

        states_values_loss = F.mse_loss(state_value, target_state_value)
        self.value_optimizer.zero_grad()
        states_values_loss.backward()
        self.value_optimizer.step()
        # target network soft update
        self._soft_update(self.value_target, self.value_network)

    def _soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

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
        print(self.name + ": loaded model")
