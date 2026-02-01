import torch
import threading
import time


def update_loop(agent, batch_size):
    while True:
        if len(agent.replay) >= batch_size:
            samples = agent.replay.sample(batch_size)
            agent.update(samples)
            time.sleep(0.001)
        else:
            time.sleep(0.01)  # tiny sleep to avoid busy-waiting


def agent_worker(agent_ctor, agent_kwargs, conn, batch_size, load_path=None, start_episode=0):
    agent = agent_ctor(**agent_kwargs)
    if start_episode >= 0:
        agent.load_model(load_path + f"ep{start_episode}/")
        agent.steps_done = 500 * (start_episode + 1)

    istate = torch.zeros(agent.internal_state_dim)
    prev_istate = torch.zeros(agent.internal_state_dim)

    threading.Thread(
        target=update_loop,
        args=(agent, batch_size),
        daemon=True
    ).start()

    while True:
        msg = conn.recv()

        if msg["cmd"] == "step":
            obs = msg["obs"]
            transition = msg.get("transition", None)

            # Add transition if available
            if transition is not None:
                agent.replay.add((
                    torch.as_tensor(transition[0], dtype=torch.float32, device=agent.device),
                    torch.as_tensor(transition[1], dtype=torch.long, device=agent.device),
                    torch.as_tensor(transition[2], dtype=torch.float32, device=agent.device),
                    torch.as_tensor(transition[3], dtype=torch.float32, device=agent.device),
                    torch.as_tensor(transition[4], dtype=torch.float32, device=agent.device),
                    prev_istate.clone().to(agent.device)
                ))
                agent.update_value(istate.clone(), transition[2], transition[3], transition[4])
                done = transition[5]
                prev_istate = istate
            else:
                prev_istate = torch.zeros(agent.internal_state_dim)
                done = False

            if not done:
                # Compute action for current state
                action, istate = agent.act(obs, istate)
                conn.send(action)

        elif msg["cmd"] == "save":
            agent.save_model(msg["path"])

        elif msg["cmd"] == "close":
            break
