import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Critic(nn.Module):
    def __init__(self, env, latent_size=64):
        super().__init__()
        self.fc1 = nn.Linear(
            np.prod(env.single_observation_space.shape)
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def save(self, run_name=None, path=None):
        if path:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self.state_dict(), f"{run_name}/models/critic.pt")
