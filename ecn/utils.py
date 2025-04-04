from collections import namedtuple
import torch
from torch.utils.data import Dataset
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from typing import NamedTuple

# Define data structures outside the class
class TensorTrajectory:
    def __init__(self, latent_states, actions, latent_next_states):
        self.latent_states = latent_states
        self.actions = actions
        self.latent_next_states = latent_next_states

class TrajectoryReplayBufferSamples(NamedTuple):
    # Fields from stable_baselines3.common.type_aliases.ReplayBufferSamples
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    # Added fields
    trajectory: TensorTrajectory
    latent_state: torch.Tensor
    latent_next_state: torch.Tensor

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
    "ExplorationBufferElement", "latent_state action latent_next_state next_state"
)

ExplorationBufferSamples = namedtuple(
    "ExplorationBufferSamples", "latent_states actions latent_next_states next_states"
)

class ExplorationBuffer:
    def __init__(self, maxlen=5000):
        self.maxlen = maxlen
        self.pos = 0
        self.full = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.latent_states = None
        self.actions = None
        self.latent_next_states = None
        self.next_states = None

    def _initialize_buffers(self, latent_state, action, latent_next_state, next_state):
        latent_size = latent_state.shape[0]
        action_size = action.shape[0]
        next_state_size = next_state.shape[0]
        
        self.latent_states = torch.zeros((self.maxlen, latent_size), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.maxlen, action_size), dtype=torch.float32, device=self.device)
        self.latent_next_states = torch.zeros((self.maxlen, latent_size), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((self.maxlen, next_state_size), dtype=torch.float32, device=self.device)

    def add(self, latent_state, action, latent_next_state, next_state):
        if self.latent_states is None:
            self._initialize_buffers(latent_state, action, latent_next_state, next_state)
        
        latent_state = torch.as_tensor(latent_state, device=self.device)
        action = torch.as_tensor(action, device=self.device)
        latent_next_state = torch.as_tensor(latent_next_state, device=self.device)
        next_state = torch.as_tensor(next_state, device=self.device)
        
        self.latent_states[self.pos] = latent_state
        self.actions[self.pos] = action
        self.latent_next_states[self.pos] = latent_next_state
        self.next_states[self.pos] = next_state
        
        self.pos = (self.pos + 1) % self.maxlen
        if not self.full and self.pos == 0:
            self.full = True
            
    def __len__(self):
        return self.maxlen if self.full else self.pos
        
    def get_last_k_elements(self, k):
        if len(self) == 0:
            return None
            
        if self.full:
            if k >= self.maxlen:
                indices = torch.arange(self.pos, self.pos + self.maxlen) % self.maxlen
            else:
                indices = torch.arange(self.pos - k, self.pos) % self.maxlen
        else:
            k = min(k, self.pos)
            indices = torch.arange(self.pos - k, self.pos) % self.maxlen
            
        return ExplorationBufferSamples(
            self.latent_states[indices],
            self.actions[indices],
            self.latent_next_states[indices],
            self.next_states[indices]
        )
    
    def sample(self, k):
        if len(self) == 0:
            return None
            
        upper_bound = self.maxlen if self.full else self.pos
        indices = torch.randint(0, upper_bound, (k,), device=self.device)
        
        return ExplorationBufferSamples(
            self.latent_states[indices],
            self.actions[indices],
            self.latent_next_states[indices],
            self.next_states[indices]
        )

class TrajectoryReplayBuffer(ReplayBuffer):
    """
    Extension of StableBaselines3 ReplayBuffer that stores trajectory history
    and latent states directly as tensors for efficient sampling.
    Assumes n_envs=1.
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
        assert n_envs == 1, "TrajectoryReplayBuffer currently supports n_envs=1 only"

        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            handle_timeout_termination=handle_timeout_termination,
            n_envs=n_envs
        )

        self.trajectory_length = trajectory_length
        self.latent_size = latent_size
        action_dim = action_space.shape[0]

        # Pre-allocate tensor storage for latent states
        self.latent_states = torch.zeros((buffer_size, latent_size), dtype=torch.float32, device=device)
        self.latent_next_states = torch.zeros((buffer_size, latent_size), dtype=torch.float32, device=device)

        # Pre-allocate tensor storage for trajectory snapshots
        self.trajectory_latent_states = torch.zeros((buffer_size, trajectory_length, latent_size), dtype=torch.float32, device=device)
        self.trajectory_actions = torch.zeros((buffer_size, trajectory_length, action_dim), dtype=torch.float32, device=device)
        self.trajectory_latent_next_states = torch.zeros((buffer_size, trajectory_length, latent_size), dtype=torch.float32, device=device)

        # Rolling buffer for the current trajectory
        self.current_trajectory_states = torch.zeros((trajectory_length, latent_size), dtype=torch.float32, device=device)
        self.current_trajectory_actions = torch.zeros((trajectory_length, action_dim), dtype=torch.float32, device=device)
        self.current_trajectory_next_states = torch.zeros((trajectory_length, latent_size), dtype=torch.float32, device=device)
        self.current_trajectory_len = 0


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
        Add a new experience and its trajectory context to the buffer.

        Args:
            obs: Current observation (shape: (n_envs, *obs_shape))
            next_obs: Next observation (shape: (n_envs, *obs_shape))
            action: Action taken (shape: (n_envs, *action_shape))
            reward: Reward received (shape: (n_envs,))
            done: Whether the episode is done (shape: (n_envs,))
            infos: Additional information (list of dicts)
            encoded_obs: Encoded current observation (latent state) (shape: (n_envs, latent_size))
            encoded_next_obs: Encoded next observation (latent next state) (shape: (n_envs, latent_size))
        """
        # Store the standard experience data
        # Note: self.pos is the index *before* adding the data
        current_index = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        # After super().add, self.pos points to the *next* available slot

        if encoded_obs is not None and encoded_next_obs is not None:
            # Ensure inputs are tensors on the correct device
            lat_state = torch.as_tensor(encoded_obs, device=self.device, dtype=torch.float32).squeeze(0) # Assuming n_envs=1
            lat_next_state = torch.as_tensor(encoded_next_obs, device=self.device, dtype=torch.float32).squeeze(0) # Assuming n_envs=1
            act_tensor = torch.as_tensor(action, device=self.device, dtype=torch.float32).squeeze(0) # Assuming n_envs=1

            if lat_state.shape[0] != self.latent_size:
                 raise ValueError(f"encoded_obs has wrong dimension: {lat_state.shape[0]}, expected {self.latent_size}")
            if lat_next_state.shape[0] != self.latent_size:
                 raise ValueError(f"encoded_next_obs has wrong dimension: {lat_next_state.shape[0]}, expected {self.latent_size}")
            if act_tensor.shape[0] != self.action_space.shape[0]:
                 raise ValueError(f"action has wrong dimension: {act_tensor.shape[0]}, expected {self.action_space.shape[0]}")


            # Store latent states for the current step
            self.latent_states[current_index] = lat_state
            self.latent_next_states[current_index] = lat_next_state

            # Update the current trajectory rolling buffer
            if self.current_trajectory_len < self.trajectory_length:
                idx = self.current_trajectory_len
                self.current_trajectory_states[idx] = lat_state
                self.current_trajectory_actions[idx] = act_tensor
                self.current_trajectory_next_states[idx] = lat_next_state
                self.current_trajectory_len += 1
            else:
                # Roll buffer to make space for the new element at the end
                self.current_trajectory_states = torch.roll(self.current_trajectory_states, shifts=-1, dims=0)
                self.current_trajectory_actions = torch.roll(self.current_trajectory_actions, shifts=-1, dims=0)
                self.current_trajectory_next_states = torch.roll(self.current_trajectory_next_states, shifts=-1, dims=0)
                # Add new element at the end
                self.current_trajectory_states[-1] = lat_state
                self.current_trajectory_actions[-1] = act_tensor
                self.current_trajectory_next_states[-1] = lat_next_state

            # Create and store the trajectory snapshot for this step
            snapshot_states = self.current_trajectory_states
            snapshot_actions = self.current_trajectory_actions
            snapshot_next_states = self.current_trajectory_next_states

            if self.current_trajectory_len < self.trajectory_length:
                # Pad at the beginning if the trajectory is not full yet
                pad_len = self.trajectory_length - self.current_trajectory_len
                pad_states = torch.zeros((pad_len, self.latent_size), dtype=torch.float32, device=self.device)
                pad_actions = torch.zeros((pad_len, self.action_space.shape[0]), dtype=torch.float32, device=self.device)
                pad_next_states = torch.zeros((pad_len, self.latent_size), dtype=torch.float32, device=self.device)

                snapshot_states = torch.cat((pad_states, self.current_trajectory_states[:self.current_trajectory_len]), dim=0)
                snapshot_actions = torch.cat((pad_actions, self.current_trajectory_actions[:self.current_trajectory_len]), dim=0)
                snapshot_next_states = torch.cat((pad_next_states, self.current_trajectory_next_states[:self.current_trajectory_len]), dim=0)

            self.trajectory_latent_states[current_index] = snapshot_states
            self.trajectory_actions[current_index] = snapshot_actions
            self.trajectory_latent_next_states[current_index] = snapshot_next_states

            # Reset trajectory if episode ends (use done before squeeze)
            if done.any(): # Assuming n_envs=1, done is (1,)
                self.current_trajectory_len = 0
                # Optionally zero out the rolling buffer tensors
                # self.current_trajectory_states.zero_()
                # self.current_trajectory_actions.zero_()
                # self.current_trajectory_next_states.zero_()


    def sample(self, batch_size, env=None):
        """
        Sample experiences from the buffer including trajectory and latent states.

        Args:
            batch_size: Number of samples to draw
            env: Optional environment for VecNormalize normalization

        Returns:
            TrajectoryReplayBufferSamples object
        """
        # Sample indices using the parent class logic (handles full/not full cases)
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound == 0: # Handle empty buffer case
             # Return empty tensors or raise error? Let's return empty tensors matching expected structure
             action_dim = self.action_space.shape[0]
             obs_shape = self.observation_space.shape
             empty_obs = torch.zeros((0, *obs_shape), device=self.device)
             empty_action = torch.zeros((0, action_dim), device=self.device)
             empty_reward = torch.zeros((0, 1), device=self.device) # rewards shape is (batch_size, n_envs)
             empty_done = torch.zeros((0, 1), dtype=torch.long, device=self.device) # dones shape is (batch_size, n_envs)
             empty_latent = torch.zeros((0, self.latent_size), device=self.device)
             empty_traj_latent = torch.zeros((0, self.trajectory_length, self.latent_size), device=self.device)
             empty_traj_action = torch.zeros((0, self.trajectory_length, action_dim), device=self.device)

             empty_tensor_traj = TensorTrajectory(empty_traj_latent, empty_traj_action, empty_traj_latent)

             return TrajectoryReplayBufferSamples(
                 observations=empty_obs,
                 actions=empty_action,
                 next_observations=empty_obs,
                 dones=empty_done,
                 rewards=empty_reward,
                 trajectory=empty_tensor_traj,
                 latent_state=empty_latent,
                 latent_next_state=empty_latent,
             )

        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        # Get standard replay samples using parent's private method
        replay_data = self._get_samples(batch_inds, env=env)

        # Sample corresponding latent states and trajectories
        latent_state = self.latent_states[batch_inds]
        latent_next_state = self.latent_next_states[batch_inds]
        traj_states = self.trajectory_latent_states[batch_inds]
        traj_actions = self.trajectory_actions[batch_inds]
        traj_next_states = self.trajectory_latent_next_states[batch_inds]

        trajectory_tensors = TensorTrajectory(traj_states, traj_actions, traj_next_states)

        # Combine standard samples with trajectory and latent states
        samples = TrajectoryReplayBufferSamples(
            observations=replay_data.observations,
            actions=replay_data.actions,
            next_observations=replay_data.next_observations,
            dones=replay_data.dones,
            rewards=replay_data.rewards,
            trajectory=trajectory_tensors,
            latent_state=latent_state,
            latent_next_state=latent_next_state,
        )
        return samples
