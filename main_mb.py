from pettingzoo.sisl import pursuit_v4
import numpy as np
import torch
from matplotlib import pyplot as plt
from multiprocessing import Process, Pipe, set_start_method

from model_based.agent import MBAgent
from model_based.agent_worker import agent_worker
import time
from functools import reduce
# ==== Config ==== #
N_AGENTS = 8            # 8 is default for pursuit
MAX_EPISODES = 1_000_001
MAX_STEPS = 64         # 500 is default for pursuit
BATCH_SIZE = 16
rewards_history_path = f'mb_rewards_history{N_AGENTS}agents-pursuit64.npy'
saved_model_path = f"model_based/saved_models{N_AGENTS}/"
plot_path = f"mb_{N_AGENTS}agents-pursuit64.png"
start_episode = -1      # -1 to start without loading models
# ================ #

def main():
    set_start_method("spawn", force=True)
    # --- Environment ---
    env = pursuit_v4.parallel_env(n_pursuers=N_AGENTS, max_cycles=MAX_STEPS, render_mode="none")
    env.reset(seed=42)

    agent_conns = {}
    agent_procs = {}

    for name in env.agents:
        parent_conn, child_conn = Pipe()
        print(name, reduce(lambda x, y: x*y, env.observation_space(name).shape))
        p = Process(
            target=agent_worker,
            args=(
                MBAgent,
                dict(
                    name=name,
                    obs_dim=reduce(lambda x, y: x*y, env.observation_space(name).shape),
                    act_dim=env.action_space(name).n,
                    eps_end=0.0001,
                    eps_decay=25_000,
                ),
                child_conn,
                BATCH_SIZE,
                saved_model_path,
                start_episode
            ),
        )

        p.start()
        agent_conns[name] = parent_conn
        agent_procs[name] = p

    print("cpu")
    print("Agents' processes: ", agent_procs)
    rewards_history = []
    if start_episode >= 0:
        rewards_history = np.load(rewards_history_path).tolist()
        rewards_history = rewards_history[:start_episode + 1]
        print("loaded rewards history", len(rewards_history), rewards_history[-3:])

    for episode in range(start_episode + 1, MAX_EPISODES):
        t0 = time.perf_counter()
        obs, _ = env.reset()
        total_reward = 0
        for name, conn in agent_conns.items():
            conn.send({
                "cmd": "step",
                "obs": obs[name].flatten(),
                "transition": None
            })

        while env.agents:
            actions = {name: conn.recv() for name, conn in agent_conns.items()}
            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            for name, conn in agent_conns.items():
                conn.send({
                    "cmd": "step",
                    "obs": obs[name].flatten(),
                    "transition": (
                        np.array(obs[name].flatten(), dtype=np.float32),
                        actions[name],
                        rewards[name],
                        np.array(next_obs[name].flatten(), dtype=np.float32),
                        np.array(terminations[name], dtype=np.float32),
                        terminations[name] or truncations[name],
                    )
                })
            obs = next_obs
            total_reward += sum(rewards.values())

        print("episode", episode, "reward:", total_reward)
        t1 = time.perf_counter()
        print("time: ", t1 - t0)
        rewards_history.append(total_reward)
        if (episode + 1) % 25 == 0:
            # plotting rolling avg rewards of agent 0
            plt.clf()
            plt.plot(rewards_history, '.', c='blue')
            rolling_avg = np.convolve(rewards_history, np.ones(100), 'valid') / 100
            plt.plot(range(100, len(rolling_avg) + 100), rolling_avg, c='red')
            ax = plt.gca()
            # ax.set_ylim([None, 0])
            plt.savefig(plot_path)
        if episode % 1000 == 0:
            # saving data for later
            np.save(rewards_history_path, rewards_history)
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