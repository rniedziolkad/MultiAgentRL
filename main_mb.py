from pettingzoo.mpe import simple_spread_v3
from model_based.agent import MBAgent
from model_based.replay_buffer import ReplayBuffer
import numpy as np
from matplotlib import pyplot as plt
import torch

N_AGENTS = 4

MAX_EPISODES = 100_001
MAX_STEPS = 25
BATCH_SIZE = 32

env = simple_spread_v3.parallel_env(N=N_AGENTS, max_cycles=MAX_STEPS, render_mode="none")
env.reset(seed=42)
agents = [MBAgent(name, env.observation_space(name).shape[0], env.action_space(name).n, n_agents=N_AGENTS)
          for name in env.agents]

print(agents)
print("cuda" if torch.cuda.is_available() else "cpu")
rewards_history = []
for episode in range(MAX_EPISODES):
    obs, _ = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        actions = {}
        for agent in agents:
            action = agent.act(obs[agent.name])
            actions[agent.name] = action

        next_obs, rewards, terminations, truncations, _ = env.step(actions)
        for agent in agents:
            agent.replay.add((
                torch.as_tensor(obs[agent.name], device=agent.device, dtype=torch.float32),
                actions[agent.name],
                rewards[agent.name],
                torch.as_tensor(next_obs[agent.name], device=agent.device, dtype=torch.float32)
            ))

        obs = next_obs
        total_reward += sum(rewards.values())

        # Update networks after enough samples collected
        for agent in agents:
            if len(agent.replay) >= BATCH_SIZE:
                samples = agent.replay.sample(BATCH_SIZE)
                agent.update(samples)
                agent.update_value(obs[agent.name], rewards[agent.name], next_obs[agent.name])

    print("episode", episode, "reward:", total_reward)
    rewards_history.append(total_reward)
    if (episode + 1) % 50 == 0:
        # plotting rolling avg rewards of agent 0
        avg_rewards = np.sum(rewards_history[-100:]) / len(rewards_history[-100:])
        plt.clf()
        plt.scatter(range(len(rewards_history)), rewards_history)
        rolling_avg = np.convolve(rewards_history, np.ones(100), 'valid') / 100
        plt.plot(range(100, len(rolling_avg) + 100), rolling_avg, c='red')
        ax = plt.gca()
        ax.set_ylim([None, 0])
        plt.savefig(f"mb{N_AGENTS}agents.png")

    if (episode + 1) % 500 == 0:
        # saving data for later
        torch.save(rewards_history, f'mb_rewards_history{N_AGENTS}agents.pth')
    if episode % 10_000 == 0:
        for agent in agents:
            agent.save_model(f"model_based/saved_models{N_AGENTS}/ep"+str(episode)+"/")

env.close()
