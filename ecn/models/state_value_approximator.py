import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StateValueApproximator(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod(),
            512,
        )
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def save(self, run_name=None, path=None):
        if path:
            torch.save(self.state_dict(), path)
        else:
            torch.save(
                self.state_dict(), f"{run_name}/models/state_value_approximator.pt"
            )
