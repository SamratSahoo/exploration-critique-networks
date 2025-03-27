import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Actor by default works in latent space
class Actor(nn.Module):
    def __init__(self, env, latent_size=64):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

    def save(self, run_name=None, path=None):
        if path:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self, f"{run_name}/models/actor.pt")
