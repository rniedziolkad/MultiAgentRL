import torch


def agent_worker(agent_ctor, agent_kwargs, conn, batch_size):
    """
    One agent per process
    """
    agent = agent_ctor(**agent_kwargs)

    while True:
        msg = conn.recv()

        if msg["cmd"] == "act":
            obs = msg["obs"]
            action = agent.act(obs)
            conn.send(action)

        elif msg["cmd"] == "store_and_update":
            transition = msg["transition"]
            agent.replay.add((
                torch.as_tensor(transition[0], dtype=torch.float32, device=agent.device),
                torch.as_tensor(transition[1], dtype=torch.long, device=agent.device),
                torch.as_tensor(transition[2], dtype=torch.float32, device=agent.device),
                torch.as_tensor(transition[3], dtype=torch.float32, device=agent.device)
            ))

            if len(agent.replay) >= batch_size:
                samples = agent.replay.sample(batch_size)
                agent.update(samples)
                agent.update_value(
                    transition[0],
                    transition[2],
                    transition[3],
                )

        elif msg["cmd"] == "save":
            agent.save_model(msg["path"])

        elif msg["cmd"] == "close":
            break
