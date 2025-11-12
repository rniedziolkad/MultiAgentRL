import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from maddpg.model import Actor, Critic
import numpy as np
import random


class MADDPGAgent:
    def __init__(self, name, obs_dim, act_dim, n_agents, gamma=0.95, tau=0.001):
        self.name = name
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        critic_input_dim = (obs_dim + act_dim) * n_agents
        self.critic = Critic(critic_input_dim).to(self.device)
        self.critic_target = Critic(critic_input_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0001)

    def act(self, obs, explore=True):
        with torch.no_grad():
            obs_tensor = torch.tensor(np.array(obs), device=self.device)
            logits = self.actor(obs_tensor)
            # probs = F.gumbel_softmax(logits, tau=0.05, hard=True)
            # return torch.argmax(probs).item()

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            if explore:
                return dist.sample().item()
            else:
                return torch.argmax(probs).item()

    def update(self, samples, all_agents):
        states, actions, rewards, next_states = samples

        batch_size = len(states)
        device = self.device

        # ======== Critic Update =========
        # Build next actions for all agents using target actors
        # For each agent, batch next_states and get target_actions from target actor
        target_actions = []
        next_states_list = []

        for agent in all_agents:
            # batch next states for this agent: [batch_size, obs_dim]
            agent_next_states = torch.tensor(
                np.array([next_states[i][agent.name] for i in range(batch_size)]),
                device=device, dtype=torch.float32
            )
            with torch.no_grad():
                logits = agent.actor_target(agent_next_states)
                action = F.gumbel_softmax(logits, tau=0.05, hard=True)
            target_actions.append(action)

            # batch states for concatenation
            next_states_list.append(agent_next_states)

        # concatenate all agents' next states and actions along last dim
        next_states_concat = torch.cat(next_states_list, dim=1)  # [batch_size, total_state_dim]
        target_actions_concat = torch.cat(target_actions, dim=1)  # [batch_size, total_action_dim]

        # Compute target Q values
        target_actions_concat = target_actions_concat.view(target_actions_concat.size(0), -1)
        critic_target_input = torch.cat([next_states_concat, target_actions_concat], dim=1)
        with torch.no_grad():
            target_q = self.critic_target(critic_target_input)
            reward = torch.tensor([r[self.name] for r in rewards], device=device, dtype=torch.float32).unsqueeze(1)
            update_target = reward + self.gamma * target_q

        # Build current critic input
        states_list = []
        actions_list = []

        for agent in all_agents:
            agent_states = torch.tensor(
                np.array([states[i][agent.name] for i in range(batch_size)]),
                device=device, dtype=torch.float32
            )
            agent_actions = torch.tensor(
                np.array([actions[i][agent.name] for i in range(batch_size)]),
                device=device, dtype=torch.long
            )
            agent_actions_onehot = F.one_hot(agent_actions, self.act_dim).float()
            states_list.append(agent_states)
            actions_list.append(agent_actions_onehot)

        states_concat = torch.cat(states_list, dim=1)  # [batch_size, total_state_dim]
        actions_concat = torch.cat(actions_list, dim=1)  # [batch_size, total_action_dim]

        critic_input = torch.cat([states_concat, actions_concat], dim=1)
        current_q = self.critic(critic_input)

        critic_loss = F.mse_loss(current_q, update_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ======== Actor Update =========
        agent_obs = torch.tensor(
            np.array([s[self.name] for s in states]),
            device=device,
            dtype=torch.float32
        )
        action_logits = self.actor(agent_obs)
        agent_actions = F.gumbel_softmax(action_logits, tau=1.0, hard=True)

        # Build new joint actions with current agent's action from policy and others fixed
        actions_for_actor = []
        for agent in all_agents:
            if agent.name == self.name:
                actions_for_actor.append(agent_actions)
            else:
                other_actions = torch.tensor(
                    np.array([actions[i][agent.name] for i in range(batch_size)]),
                    device=device,
                    dtype=torch.long
                )
                other_actions_onehot = F.one_hot(other_actions, self.act_dim).float()
                actions_for_actor.append(other_actions_onehot)

        new_actions_concat = torch.cat(actions_for_actor, dim=1)
        actor_input = torch.cat([states_concat, new_actions_concat], dim=1)
        actor_q = self.critic(actor_input)
        actor_loss = -actor_q.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if random.randint(1, 200) == 1:
            print(self.name + " critic loss:", critic_loss)
            print(self.name + " actor loss:", actor_loss)
        # ======== Target Network Soft Updates =========
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def __repr__(self):
        return self.name + "[obs: " + str(self.obs_dim) + " act: " + str(self.act_dim) + "]"

    def save_model(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.actor.state_dict(), dir_path+self.name+"_actor.pth")
        torch.save(self.critic.state_dict(), dir_path+self.name+"_critic.pth")

    def load_model(self, dir_path):
        self.actor.load_state_dict(torch.load(dir_path+self.name+"_actor.pth"))
        self.critic.load_state_dict(torch.load(dir_path+self.name+"+_critic.pth"))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())