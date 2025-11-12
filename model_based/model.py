import torch.nn as nn


class EnvironmentModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.layer1 = nn.Linear(obs_dim, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.next_states_layer = nn.Linear(128, act_dim * obs_dim)
        self.reward_layer = nn.Linear(128, act_dim)
        # self.final_layer = nn.Linear(128, act_dim)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        next_states = self.next_states_layer(x)
        rewards = self.reward_layer(x)
        # final = nn.functional.sigmoid(self.final_layer(x))
        return next_states, rewards     # , final


# Value Network V(state) --- "how valuable being in given state is"
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)
