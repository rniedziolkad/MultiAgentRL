from pettingzoo.sisl import pursuit_v4
from model_based.agent import MBAgent
import numpy as np
from matplotlib import pyplot as plt
import torch
import time
from functools import reduce

N_AGENTS = 8

MAX_EPISODES = 1_000_001
MAX_STEPS = 500
BATCH_SIZE = 32

env = pursuit_v4.parallel_env(n_pursuers=N_AGENTS, max_cycles=MAX_STEPS, render_mode="none")
env.reset(seed=42)
agents = [MBAgent(name, reduce(lambda x, y: x*y, env.observation_space(name).shape), env.action_space(name).n,
                  eps_end=0.0001, eps_decay=10000)
          for name in env.agents]

print(agents)
print("cpu")
rewards_history = []
for episode in range(MAX_EPISODES):
    obs, _ = env.reset()
    for a in obs.keys():
        obs[a] = obs[a].flatten()
    total_reward = 0
    t0 = time.perf_counter()

    for step in range(MAX_STEPS):
        actions = {}
        for agent in agents:
            action = agent.act(obs[agent.name])
            actions[agent.name] = action

        next_obs, rewards, terminations, truncations, _ = env.step(actions)
        for a in next_obs.keys():
            next_obs[a] = next_obs[a].flatten()
        for agent in agents:
            agent.replay.add((
                torch.as_tensor(obs[agent.name], device=agent.device, dtype=torch.float32),
                torch.tensor(actions[agent.name], device=agent.device, dtype=torch.long),
                torch.tensor(rewards[agent.name], device=agent.device, dtype=torch.float32),
                torch.as_tensor(next_obs[agent.name], device=agent.device, dtype=torch.float32)
            ))

        # Update networks after enough samples collected
        for agent in agents:
            if len(agent.replay) >= BATCH_SIZE:
                samples = agent.replay.sample(BATCH_SIZE)
                agent.update(samples)
                agent.update_value(obs[agent.name], rewards[agent.name], next_obs[agent.name])

        obs = next_obs
        total_reward += sum(rewards.values())

    print("episode", episode, "reward:", total_reward)
    t1 = time.perf_counter()
    print("time:", t1 - t0)
    rewards_history.append(total_reward)
    if (episode + 1) % 25 == 0:
        # plotting rolling avg rewards of agent 0
        avg_rewards = np.sum(rewards_history[-100:]) / len(rewards_history[-100:])
        plt.clf()
        plt.scatter(range(len(rewards_history)), rewards_history)
        rolling_avg = np.convolve(rewards_history, np.ones(100), 'valid') / 100
        plt.plot(range(100, len(rolling_avg) + 100), rolling_avg, c='red')
        ax = plt.gca()
        ax.set_ylim([None, 0])
        plt.savefig(f"mb_target{N_AGENTS}agents.png")
        # saving data for later
        torch.save(rewards_history, f'mb_rewards_history_target{N_AGENTS}agents.pth')
    if episode % 500 == 0:
        for agent in agents:
            agent.save_model(f"model_based/saved_models{N_AGENTS}/ep"+str(episode)+"/")

env.close()
