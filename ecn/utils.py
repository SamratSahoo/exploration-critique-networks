from collections import deque, namedtuple
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

class RolloutDataset(Dataset):
    def __init__(self, states_np, returns_np, next_states_np, episodic_returns):
        """
        Custom PTyorch dataset to store rollout data.

        Args:
        - states_np (numpy.ndarray): Current states
        - returns_np (numpy.ndarray): Corresponding returns
        - next_states_np (numpy.ndarray): Next states
        - episodic_returns (numpy.ndarray): Next states
        """
        self.states = torch.tensor(states_np, dtype=torch.float32)
        self.returns = torch.tensor(returns_np, dtype=torch.float32)
        self.next_states = torch.tensor(next_states_np, dtype=torch.float32)
        self.episodic_returns = torch.tensor(episodic_returns, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.returns[idx],
            self.next_states[idx],
            self.episodic_returns[idx],
        )

ExplorationBufferElement = namedtuple(
    "ExplorationBufferElement", "latent_state action latent_next_state"
)
class ExplorationBuffer:
    def __init__(self, maxlen=5000):
        self.buffer = deque(maxlen=maxlen)

    def add(self, latent_state, action, latent_next_state):
        self.buffer.append(
            ExplorationBufferElement(latent_state, action, latent_next_state)
        )
        
    def get_last_k_elements(self, k):
        return list(self.buffer)[max(len(self.buffer) - k, 0):]
