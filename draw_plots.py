import torch
import matplotlib.pyplot as plt
import pandas as pd

window_size = 5_000
maddpg_reward_history = torch.load('maddpg_rewards_history8agents.pth')
# mb_reward_history = torch.load('mb_rewards_history_no-target4agents.pth')
# mb_reward_history2 = torch.load('mb_rewards_history_target4agents.pth')
mb_reward_history = torch.load('mb_rewards_history_par8agents.pth')
# dqn_reward_history = torch.load("dqn_rewards_history3agents.pth")

maddpg_reward_series0 = pd.Series(maddpg_reward_history)
maddpg_rolling_average0 = maddpg_reward_series0.rolling(window=window_size).mean()

# dqn_reward_series0 = pd.Series(dqn_reward_history)
# dqn_rolling_average0 = dqn_reward_series0.rolling(window=window_size).mean()

mb_reward_series0 = pd.Series(mb_reward_history)
mb_rolling_average0 = mb_reward_series0.rolling(window=window_size).mean()

# mb2_reward_series0 = pd.Series(mb_reward_history2)
# mb2_rolling_average0 = mb2_reward_series0.rolling(window=window_size).mean()

# mbP_reward_series0 = pd.Series(mb_reward_historyP)
# mbP_rolling_average0 = mbP_reward_series0.rolling(window=window_size).mean()

plt.plot(maddpg_rolling_average0, label="MADDPG")
# plt.plot(dqn_rolling_average0, label="DQN")
plt.plot(mb_rolling_average0[:300000], label="MB")
# plt.plot(mb2_rolling_average0, label="MB_target")
# plt.plot(mbP_rolling_average0, label="MB_parallel")

plt.xlabel('Episode')
plt.ylabel('Average Total Reward')
plt.legend()
# plt.xticks(np.arange(0, 90_001, 10_000))
plt.savefig("ss8_learning_results.pdf")
# plt.ylim(bottom=-400)
# plt.xlim(left=0, right=30_000)
plt.show()


