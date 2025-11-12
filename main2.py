from pettingzoo.mpe import simple_spread_v3
from independent_dqn.agent import DQNAgent
import numpy as np
from matplotlib import pyplot as plt
import torch

N_AGENTS = 3

MAX_EPISODES = 100_001
MAX_STEPS = 25
BATCH_SIZE = 2

env = simple_spread_v3.parallel_env(N=N_AGENTS, max_cycles=MAX_STEPS, render_mode="none")
env.reset(seed=42)
agents = [DQNAgent(name, env.observation_space(name).shape[0], env.action_space(name).n)
          for name in env.agents]

print(agents)
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
            agent.replay.add((obs[agent.name], actions[agent.name], rewards[agent.name], next_obs[agent.name]))

        obs = next_obs
        total_reward += sum(rewards.values())

        # Update networks after enough samples collected
        for agent in agents:
            if len(agent.replay) >= BATCH_SIZE:
                samples = agent.replay.sample(BATCH_SIZE)
                agent.update(samples)

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
        plt.savefig(f"dqn{N_AGENTS}agents.png")

    if (episode + 1) % 500 == 0:
        # saving data for later
        torch.save(rewards_history, f'dqn_rewards_history{N_AGENTS}agents.pth')
    if episode % 10_000 == 0:
        for agent in agents:
            agent.save_model(f"independent_dqn/saved_models{N_AGENTS}/ep"+str(episode)+"/")

env.close()
