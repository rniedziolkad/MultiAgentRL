import math
import os
import torch
import torch.optim as optim
from model_based.model import ValueNetwork, EnvironmentModel, HistoryEncoder, HistoryDecoder
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

        self.history_encoder = HistoryEncoder(self.internal_state_dim, obs_dim).to(self.device)
        self.history_decoder = HistoryDecoder(self.internal_state_dim, obs_dim).to(self.device)
        self.environment_model = EnvironmentModel(self.internal_state_dim, act_dim).to(self.device)

        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.0001)
        self.encoder_optimizer = optim.Adam(self.history_encoder.parameters(), lr=0.0001)
        self.decoder_optimizer = optim.Adam(self.history_decoder.parameters(), lr=0.0001)
        self.environment_optimizer = optim.Adam(self.environment_model.parameters(), lr=0.0001)

        self.replay = ReplayBuffer(64)
        self.steps_done = 0

    def act(self, obs, internal_state, explore=True):
        with torch.inference_mode():
            obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            istate = self.history_encoder(internal_state, obs_tensor)
            next_istates, rewards, finals = self.environment_model(istate)
            next_istates = next_istates.view(self.act_dim, self.internal_state_dim)
            next_states_values = self.value_network(next_istates).flatten()
            finals = torch.sigmoid(finals)
            if explore:
                eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
                self.steps_done += 1
                if np.random.random_sample() < eps:
                    return np.random.randint(0, self.act_dim), istate

            expected_returns = self.gamma * next_states_values * (1.0 - finals) + rewards
            return torch.argmax(expected_returns).item(), istate

    def update(self, samples):
        observations, actions, rewards, next_observations, finals, prev_istates = samples
        # ========  Encoder-Decoder update  ========
        istates = self.history_encoder(prev_istates, observations)
        next_istates = self.history_encoder(istates, next_observations)
        decoded_istates, decoded_next_observations = self.history_decoder(next_istates)
        autoencoder_loss = F.mse_loss(decoded_next_observations, next_observations)
        decoded_prev_istates, decoded_observations = self.history_decoder(decoded_istates)
        autoencoder_loss += F.mse_loss(decoded_observations, observations)
        autoencoder_loss += F.mse_loss(decoded_prev_istates, prev_istates)
        self.decoder_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        autoencoder_loss.backward()
        self.decoder_optimizer.step()
        self.encoder_optimizer.step()
        # ======== Environment Model update ========
        with torch.no_grad():
            istates = self.history_encoder(prev_istates, observations)
        pred_next_istates, pred_rewards, pred_finals = self.environment_model(istates)
        # reshape pred_next_istates: (B, act_dim * istate_dim) → (B, act_dim, istate_dim)
        pred_next_istates = pred_next_istates.view(-1, self.act_dim, self.internal_state_dim)
        # gather predictions for actual actions
        batch_idx = torch.arange(pred_next_istates.size(0), device=self.device)
        pred_next_istates = pred_next_istates[batch_idx, actions]
        decoded_istates, pred_next_observations = self.history_decoder(pred_next_istates)
        pred_rewards = pred_rewards[batch_idx, actions]
        pred_finals = pred_finals[batch_idx, actions]
        # compute loss and optimize environment model
        next_observations_loss = F.mse_loss(pred_next_observations, next_observations)
        reconstruction_loss = F.mse_loss(decoded_istates, istates)
        rewards_loss = F.mse_loss(pred_rewards, rewards)
        finals_loss = F.binary_cross_entropy_with_logits(pred_finals, finals)
        environment_loss = next_observations_loss + reconstruction_loss + rewards_loss + finals_loss
        self.environment_optimizer.zero_grad()
        # self.encoder_optimizer.zero_grad()
        # self.decoder_optimizer.zero_grad()
        environment_loss.backward()
        self.environment_optimizer.step()
        # self.encoder_optimizer.step()
        # self.decoder_optimizer.step()

    def update_value(self, istate, reward, next_observation, final):
        next_observation = torch.as_tensor(next_observation, dtype=torch.float32, device=self.device)
        # ======== Value Network Update ========
        state_value = self.value_network(istate)
        with torch.no_grad():
            next_istate = self.history_encoder(istate, next_observation)
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
        torch.save(self.value_target.state_dict(), dir_path+self.name+"_target.pth")
        torch.save(self.environment_model.state_dict(), dir_path+self.name+"_environment.pth")
        torch.save(self.history_encoder.state_dict(), dir_path+self.name+"_encoder.pth")
        torch.save(self.history_decoder.state_dict(), dir_path+self.name+"_decoder.pth")

    def load_model(self, dir_path):
        self.value_network.load_state_dict(torch.load(dir_path+self.name+"_value.pth"))
        self.value_target.load_state_dict(torch.load(dir_path+self.name+"_target.pth"))
        self.environment_model.load_state_dict(torch.load(dir_path+self.name+"_environment.pth"))
        self.history_encoder.load_state_dict(torch.load(dir_path+self.name+"_encoder.pth"))
        self.history_decoder.load_state_dict(torch.load(dir_path+self.name+"_decoder.pth"))
        print(self.name + ": loaded model")
