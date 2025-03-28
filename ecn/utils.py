from collections import deque, namedtuple
import torch
from torch.utils.data import Dataset
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

class TrajectoryReplayBufferSamples:
    """
    Samples class that contains both the original ReplayBufferSamples and trajectory data.
    """
    def __init__(self, replay_samples, trajectory):
        self.observations = replay_samples.observations
        self.actions = replay_samples.actions
        self.next_observations = replay_samples.next_observations
        self.dones = replay_samples.dones
        self.rewards = replay_samples.rewards
        self.trajectory = trajectory

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

class TrajectoryReplayBuffer(ReplayBuffer):
    """
    Extension of StableBaselines3 ReplayBuffer that also tracks trajectory history
    for each state in the buffer.
    """
    
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        latent_size,
        trajectory_length=50,
        handle_timeout_termination=True,
        n_envs=1,
    ):
        super().__init__(
            buffer_size=buffer_size, 
            observation_space=observation_space, 
            action_space=action_space, 
            device=device, 
            handle_timeout_termination=handle_timeout_termination, 
            n_envs=n_envs
        )
        
        self.trajectory_length = trajectory_length
        self.trajectory_buffer = deque(maxlen=buffer_size)
        self.current_trajectory = deque(maxlen=trajectory_length)
        
        action_dim = action_space.shape[0]
        self.empty_element = ExplorationBufferElement(
            torch.zeros(latent_size, device=device),
            torch.zeros(action_dim, device=device),
            torch.zeros(latent_size, device=device)
        )
    
    def add(
        self, 
        obs, 
        next_obs,
        action, 
        reward, 
        done, 
        infos, 
        encoded_obs=None, 
        encoded_next_obs=None
    ):
        """
        Add a new experience to the buffer with its trajectory history.
        
        Args:
            obs: Current observation
            next_obs: Next observation
            action: Action taken
            reward: Reward received
            done: Whether the episode is done
            infos: Additional information
            encoded_obs: Encoded current observation (latent state)
            encoded_next_obs: Encoded next observation (latent next state)
        """
        super().add(obs, next_obs, action, reward, done, infos)
        
        if encoded_obs is not None and encoded_next_obs is not None:
            self.current_trajectory.append(
                ExplorationBufferElement(
                    encoded_obs.squeeze(0) if encoded_obs.dim() > 1 else encoded_obs,
                    torch.tensor(action, dtype=torch.float32, device=self.device).squeeze(0) 
                        if hasattr(action, 'shape') and len(action.shape) > 1 else 
                        torch.tensor(action, dtype=torch.float32, device=self.device),
                    encoded_next_obs.squeeze(0) if encoded_next_obs.dim() > 1 else encoded_next_obs
                )
            )
            
            trajectory_snapshot = list(self.current_trajectory)
            
            if len(trajectory_snapshot) < self.trajectory_length:
                padding = [self.empty_element] * (self.trajectory_length - len(trajectory_snapshot))
                trajectory_snapshot = padding + trajectory_snapshot
            
            self.trajectory_buffer.append(trajectory_snapshot)
            
            if done:
                self.current_trajectory.clear()
    
    def sample(self, batch_size):
        """
        Sample experiences from the buffer and include their trajectories.
        
        Args:
            batch_size: Number of samples to draw
        
        Returns:
            TrajectoryReplayBufferSamples object with trajectory attribute
        """
        if self.full:
            batch_inds = np.random.randint(0, self.buffer_size, size=batch_size)
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        
        replay_samples = super().sample(batch_size)
        
        # If trajectory buffer is empty, create a dummy trajectory
        if len(self.trajectory_buffer) == 0:
            # Get dimensions
            latent_state_dim = self.empty_element.latent_state.shape[0]
            action_dim = self.empty_element.action.shape[0]
            
            # Create empty tensors
            latent_states = torch.zeros(
                (batch_size, self.trajectory_length, latent_state_dim), 
                device=self.device
            )
            actions = torch.zeros(
                (batch_size, self.trajectory_length, action_dim), 
                device=self.device
            )
            latent_next_states = torch.zeros(
                (batch_size, self.trajectory_length, latent_state_dim), 
                device=self.device
            )
            
            class TensorTrajectory:
                def __init__(self, latent_states, actions, latent_next_states):
                    self.latent_states = latent_states
                    self.actions = actions
                    self.latent_next_states = latent_next_states
            
            dummy_trajectory = TensorTrajectory(latent_states, actions, latent_next_states)
            return TrajectoryReplayBufferSamples(replay_samples, dummy_trajectory)
        
        # Get trajectories for sampled indices
        trajectory_batch = []
        for i in batch_inds:
            if 0 <= i < len(self.trajectory_buffer):
                trajectory_batch.append(self.trajectory_buffer[i])
            else:
                trajectory_batch.append([self.empty_element] * self.trajectory_length)
        
        # Convert list of lists to tensor
        trajectory_tensor = self._convert_trajectories_to_tensor(trajectory_batch)
        
        # Return new samples object with trajectory
        return TrajectoryReplayBufferSamples(replay_samples, trajectory_tensor)
    
    def _convert_trajectories_to_tensor(self, trajectories):
        """
        Convert a list of trajectory lists to a tensor structure.
        
        Args:
            trajectories: List of trajectory lists
        
        Returns:
            Tensor representation of trajectories
        """
        # Get dimensions from the first trajectory
        sample_trajectory = trajectories[0][0]
        latent_state_dim = sample_trajectory.latent_state.shape[0]
        action_dim = sample_trajectory.action.shape[0]
        
        batch_size = len(trajectories)
        
        # Initialize tensors
        latent_states = torch.zeros(
            (batch_size, self.trajectory_length, latent_state_dim), 
            device=self.device
        )
        actions = torch.zeros(
            (batch_size, self.trajectory_length, action_dim), 
            device=self.device
        )
        latent_next_states = torch.zeros(
            (batch_size, self.trajectory_length, latent_state_dim), 
            device=self.device
        )
        
        # Fill tensors
        for i, trajectory in enumerate(trajectories):
            for j, element in enumerate(trajectory):
                latent_states[i, j] = element.latent_state
                actions[i, j] = element.action
                latent_next_states[i, j] = element.latent_next_state
        
        # Create a new structure similar to ExplorationBufferElement but using tensors
        class TensorTrajectory:
            def __init__(self, latent_states, actions, latent_next_states):
                self.latent_states = latent_states
                self.actions = actions
                self.latent_next_states = latent_next_states
        
        return TensorTrajectory(latent_states, actions, latent_next_states)
