from pettingzoo.mpe import simple_spread_v3
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Process, Pipe, set_start_method

from model_based.agent import MBAgent
from model_based.agent_worker import agent_worker
import time
# ==== Config ==== #
N_AGENTS = 6
MAX_EPISODES = 1_000_001
MAX_STEPS = 25
BATCH_SIZE = 32
# ================ #


def main():
    set_start_method("spawn", force=True)
    # --- Environment ---
    env = simple_spread_v3.parallel_env( N=N_AGENTS, max_cycles=MAX_STEPS, render_mode="none")
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
            ),
        )

        p.start()
        agent_conns[name] = parent_conn
        agent_procs[name] = p

    print("cpu")
    print("Agents' processes: ", agent_procs)
    rewards_history = []
    for episode in range(MAX_EPISODES):
        obs, _ = env.reset()
        total_reward = 0
        t0 = time.perf_counter()

        for step in range(MAX_STEPS):
            for name, conn in agent_conns.items():
                conn.send({"cmd": "act", "obs": obs[name]})

            actions = {
                name: conn.recv()
                for name, conn in agent_conns.items()
            }

            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            for name, conn in agent_conns.items():
                conn.send({
                    "cmd": "store_and_update",
                    "transition": (
                        np.array(obs[name], dtype=np.float32),
                        actions[name],
                        rewards[name],
                        np.array(next_obs[name], dtype=np.float32),
                    ),
                })

            obs = next_obs
            total_reward += sum(rewards.values())

        print("episode", episode, "reward:", total_reward)
        t1 = time.perf_counter()
        print("time: ", t1 - t0)
        rewards_history.append(total_reward)
        if (episode + 1) % 500 == 0:
            # plotting rolling avg rewards of agent 0
            plt.clf()
            plt.scatter(range(len(rewards_history)), rewards_history)
            rolling_avg = np.convolve(rewards_history, np.ones(100), 'valid') / 100
            plt.plot(range(100, len(rolling_avg) + 100), rolling_avg, c='red')
            ax = plt.gca()
            ax.set_ylim([None, 0])
            plt.savefig(f"mb{N_AGENTS}agents.png")
            # saving data for later
            torch.save(rewards_history, f'mb_rewards_history{N_AGENTS}agents.pth')
        if episode % 10_000 == 0:
            for name, conn in agent_conns.items():
                conn.send({
                    "cmd": "save",
                    "path": f"model_based/saved_models{N_AGENTS}/ep{episode}/",
                })

    # ---- Shutdown ----
    for conn in agent_conns.values():
        conn.send({"cmd": "close"})

    for p in agent_procs.values():
        p.join()

    env.close()


if __name__ == "__main__":
    main()

