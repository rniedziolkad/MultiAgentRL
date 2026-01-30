from pettingzoo.mpe import simple_spread_v3
import numpy as np
import torch
from matplotlib import pyplot as plt
from multiprocessing import Process, Pipe, set_start_method

from model_based.agent import MBAgent
from model_based.agent_worker import agent_worker
import time
# ==== Config ==== #
N_AGENTS = 3                # 3 is default for Simple Spread
MAX_EPISODES = 1_000_001
MAX_STEPS = 25              # 25 is default for Simple Spread
BATCH_SIZE = 32
rewards_history_path = f'mb_rewards_history_par{N_AGENTS}agents.pth'
saved_model_path = f"model_based/saved_models{N_AGENTS}/"
plot_path = f"mb_par{N_AGENTS}agents.png"
START_EPISODE = 290_000
# ================ #


def main():
    set_start_method("spawn", force=True)
    # --- Environment ---
    env = simple_spread_v3.parallel_env(N=N_AGENTS, max_cycles=MAX_STEPS, render_mode="none")
    env.reset(seed=42)

    agent_conns = {}
    agent_procs = {}

    for name in env.agents:
        parent_conn, child_conn = Pipe()

        p = Process(
            target=agent_worker,
            args=(
                MBAgent,
                dict(
                    name=name,
                    obs_dim=env.observation_space(name).shape[0],
                    act_dim=env.action_space(name).n,
                    eps_end=0.0001,
                    eps_decay=10000,
                ),
                child_conn,
                BATCH_SIZE,
                saved_model_path,
                START_EPISODE
            ),
        )

        p.start()
        agent_conns[name] = parent_conn
        agent_procs[name] = p

    print("cpu")
    print("Agents' processes: ", agent_procs)
    rewards_history = []
    if START_EPISODE != 0:
        rewards_history = torch.load(rewards_history_path)
        rewards_history = rewards_history[:START_EPISODE+1]
        print("loaded rewards history", len(rewards_history), rewards_history[-3:])

    for episode in range(START_EPISODE+1, MAX_EPISODES):
        obs, _ = env.reset()
        total_reward = 0
        t0 = time.perf_counter()
        prev_obs = None

        for step in range(MAX_STEPS):
            for name, conn in agent_conns.items():
                conn.send({
                    "cmd": "step",
                    "obs": obs[name],
                    "transition": (
                        np.array(prev_obs[name], dtype=np.float32),
                        actions[name],
                        rewards[name],
                        np.array(obs[name], dtype=np.float32),
                    ) if prev_obs is not None else None
                })

            actions = {name: conn.recv() for name, conn in agent_conns.items()}
            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            prev_obs = obs
            obs = next_obs
            total_reward += sum(rewards.values())

        print("episode", episode, "reward:", total_reward)
        t1 = time.perf_counter()
        print("time: ", t1 - t0)
        rewards_history.append(total_reward)
        if (episode + 1) % 500 == 0:
            # plotting rolling avg rewards of agent 0
            plt.clf()
            plt.plot(rewards_history, '.', c ='blue')
            rolling_avg = np.convolve(rewards_history, np.ones(100), 'valid') / 100
            plt.plot(range(100, len(rolling_avg) + 100), rolling_avg, c='red')
            ax = plt.gca()
            # ax.set_ylim([None, 0])
            plt.savefig(plot_path)
        if episode % 10_000 == 0:
            # saving data for later
            torch.save(rewards_history, rewards_history_path)
            for name, conn in agent_conns.items():
                conn.send({
                    "cmd": "save",
                    "path": saved_model_path + f"ep{episode}/",
                })

    # ---- Shutdown ----
    for conn in agent_conns.values():
        conn.send({"cmd": "close"})

    for p in agent_procs.values():
        p.join()

    env.close()


if __name__ == "__main__":
    main()

