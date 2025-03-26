# Base DDPG Implementation from CleanRL: https://github.com/vwxyzjn/cleanrl
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from collections import deque, namedtuple
import math
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, Dataset
import stable_baselines3 as sb3
from sys import platform

os.environ["MUJOCO_GL"] = "glfw" if platform == "darwin" else "osmesa"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v5"
    """the environment id of the Atari game"""
    total_timesteps: int = 2000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""


args = tyro.cli(Args)
run_name = f"runs/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
os.makedirs(run_name, exist_ok=True)
os.makedirs(f"{run_name}/models", exist_ok=True)
os.makedirs(f"{run_name}/logs", exist_ok=True)
os.makedirs(f"{run_name}/videos", exist_ok=True)
os.makedirs(f"{run_name}/eval", exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
torch.set_default_device(device)

writer = SummaryWriter(log_dir=f"{run_name}/logs")


def make_env(env_id, seed, idx, capture_video, run_name, is_eval=False):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            if is_eval:
                env = gym.wrappers.RecordVideo(env, f"{run_name}/eval")
            else:
                env = gym.wrappers.RecordVideo(env, f"{run_name}/videos")
            env.recorded_frames = []
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)],
    autoreset_mode=gym.vector.vector_env.AutoresetMode.SAME_STEP,
)

ExplorationBufferElement = namedtuple(
    "ExplorationBufferElement", "latent_state action latent_next_state"
)


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

    def save(self, path=None):
        if path:
            torch.save(self.state_dict(), path)
        else:
            torch.save(
                self.state_dict(), f"{run_name}/models/state_value_approximator.pt"
            )


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

    def save(self, path=None):
        if path:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self.state_dict(), f"{run_name}/models/critic.pt")


class StateAggregationEncoder(nn.Module):
    def __init__(self, env, latent_size=64):
        super(StateAggregationEncoder, self).__init__()
        self.latent_size = latent_size

        # Input Shape is flattened state space + State Value
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.batch_norm_1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm_2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, self.latent_size)
        self.batch_norm_3 = nn.BatchNorm1d(self.latent_size)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

    def forward(self, state):
        x = self.fc1(state)
        x = self.batch_norm_1(x)
        x = self.prelu1(x)

        x = self.fc2(x)
        x = self.batch_norm_2(x)
        x = self.prelu2(x)

        x = self.fc3(x)
        x = self.batch_norm_3(x)

        return x

    def save(self, path=None):
        if path:
            torch.save(self.state_dict(), path)
        else:
            torch.save(
                self.state_dict(), f"{run_name}/models/state_aggregation_encoder.pt"
            )


class StateAggregationDecoder(nn.Module):
    def __init__(self, env, latent_size=64):
        super(StateAggregationDecoder, self).__init__()

        self.latent_size = latent_size

        self.fc1 = nn.Linear(self.latent_size, 128)
        self.batch_norm_1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.batch_norm_2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, np.array(env.single_observation_space.shape).prod())

        self.transition_head = nn.Linear(
            256, np.array(env.single_observation_space.shape).prod()
        )
        
        self.episodic_return_head = nn.Linear(
            256, np.array(env.single_observation_space.shape).prod()
        )

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

    def forward(self, latent_state):
        x = self.fc1(latent_state)
        x = self.batch_norm_1(x)
        x = self.prelu1(x)

        x = self.fc2(x)
        x = self.batch_norm_2(x)
        x = self.prelu2(x)

        current_state_pred = self.fc3(x)
        next_state_pred = self.transition_head(x)
        return current_state_pred, next_state_pred

    def save(self, path=None):
        if path:
            torch.save(self.state_dict(), path)
        else:
            torch.save(
                self.state_dict(), f"{run_name}/models/state_aggregation_decoder.pt"
            )

class StateAggregationAutoencoder(nn.Module):
    def __init__(self, env: gym.Env, latent_size=64, alpha=0.99):
        super(StateAggregationAutoencoder, self).__init__()

        self.encoder = StateAggregationEncoder(env, latent_size=latent_size)
        self.decoder = StateAggregationDecoder(env, latent_size=latent_size)
        
        # Running averages for loss scaling
        self.register_buffer("running_avg_loss_current", torch.tensor(1.0))
        self.register_buffer("running_avg_loss_next", torch.tensor(1.0))
        self.register_buffer("running_avg_return", torch.tensor(1.0))
        
        self.alpha = alpha  # Smoothing factor for EMA
    
    def forward(self, state):
        encoder_out = self.encoder(state)
        current_state_pred, next_state_pred= self.decoder(encoder_out)
        return encoder_out, current_state_pred, next_state_pred

    def update_running_average(self, current, new_value):
        return self.alpha * current + (1 - self.alpha) * new_value

    def compute_loss(
        self,
        encoder_out,
        next_state_pred,
        next_states,
        current_state_pred,
        current_state,
        episodic_return,
        autoencoder_regularization_coefficient,
    ):
        loss_current_state = F.mse_loss(current_state_pred, current_state)
        loss_next_state = F.mse_loss(next_state_pred, next_states)
        episodic_return_mean = torch.tensor(episodic_return.mean(), device=loss_current_state.device)
        with torch.no_grad():
            self.running_avg_loss_current = self.update_running_average(
                self.running_avg_loss_current, loss_current_state.detach()
            )
            self.running_avg_loss_next = self.update_running_average(
                self.running_avg_loss_next, loss_next_state.detach()
            )
            # self.running_avg_return = self.update_running_average(
            #     self.running_avg_return, episodic_return_mean.detach()
            # )
        
        # Scale losses by running averages
        scaled_loss_current = loss_current_state / (self.running_avg_loss_current + 1e-8)
        scaled_loss_next = loss_next_state / (self.running_avg_loss_next + 1e-8)
        # scaled_return = episodic_return.mean() / (self.running_avg_return + 1e-8)
        return (scaled_loss_current + scaled_loss_next)

    def save(self, path=None):
        self.encoder.save()
        self.decoder.save()

        if path:
            torch.save(self.state_dict(), path)
        else:
            torch.save(
                self.state_dict(), f"{run_name}/models/state_aggregation_autoencoder.pt"
            )


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

    def save(self, path=None):
        if path:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self, f"{run_name}/models/actor.pt")


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, length, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        positional_encoding = torch.zeros(length, embedding_size)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size)
        )

        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        positional_encoding = positional_encoding.unsqueeze(1)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, exploration_buffer_seq):
        exploration_buffer_seq += self.positional_encoding[
            : exploration_buffer_seq.size(0)
        ]
        return exploration_buffer_seq


class ExplorationBuffer:
    def __init__(self, maxlen=5000):
        self.buffer = deque(maxlen=maxlen)

    def add(self, latent_state, action, latent_next_state):
        self.buffer.append(
            ExplorationBufferElement(latent_state, action, latent_next_state)
        )
        
    def get_last_k_elements(self, k):
        return self.buffer[max(len(self.buffer) - k, 0):]


class ExplorationCritic(nn.Module):

    def __init__(
        self,
        env,
        latent_state_size=64,
        transformer_embedding_size=128,
        max_seq_len=50,
        heads=8,
    ):
        """
        Args:
            latent_state_size (int, optional): Size of latent state created from the State Aggregation Module. Defaults to 64.
            transformer_embedding_size (int, optional): Size of transformer embedding. Defaults to 128.
            max_seq_len (int, optional): Number of elements you want to consider from the exploration buffer. Defaults to 50.
            nheads (int, optional): Number of transformer heads
        """
        super(ExplorationCritic, self).__init__()
        # Embedding for state tokens
        self.state_embedding = nn.Linear(latent_state_size, transformer_embedding_size)
        # Embedding for action tokens
        self.action_embedding = nn.Linear(
            np.array(env.single_action_space.shape).prod(), transformer_embedding_size
        )
        # Embedding for exploration buffer tokens
        self.exploration_embedding = nn.Linear(
            2*latent_state_size + np.array(env.single_action_space.shape).prod(),
            transformer_embedding_size,
        )

        # Positional encoding layer for exploration buffer
        self.positional_encoding = PositionalEncoding(
            embedding_size=transformer_embedding_size, length=max_seq_len
        )

        # Encoding layer for exploration buffer
        encode_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embedding_size, nhead=heads
        )
        self.exploration_transfromer = nn.TransformerEncoder(encode_layer, num_layers=8)

        # Cross Attention layer between exploration buffer sequence and current state/action/next state sequence
        # Evaluates novelty of current exploration in reference to previous experience
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_embedding_size, nhead=heads
        )
        self.cross_attention = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.fc_out = nn.Sequential(
            nn.Linear(transformer_embedding_size, transformer_embedding_size // 2),
            nn.ReLU(),
            nn.Linear(transformer_embedding_size // 2, 1),
        )
    
    def forward(self, latent_state, action, latent_next_state, exploration_buffer_seq):
        # Embed latent state, action, latent next state,
        current_state_token = self.state_embedding(latent_state)
        action_token = self.action_embedding(action)
        next_state_token = self.state_embedding(latent_next_state)

        # Embed + add positional encoding to exploration buffer sequence
        expl_buffer_token = self.exploration_embedding(exploration_buffer_seq)
        pos_expl_buffer_token = self.positional_encoding(expl_buffer_token)

        # Create query
        query = torch.cat([current_state_token, action_token, next_state_token], dim=0)

        # Get cross attention
        cross_attention_out = self.cross_attention(query, pos_expl_buffer_token)
        aggregate_cross = cross_attention_out.mean(dim=0)
        exploration_score = self.fc_out(aggregate_cross).squeeze(-1)
        return exploration_score

    def save(self, path=None):
        if not path:
            torch.save(self.state_dict(), f"{run_name}/models/exploration_critic.pt")
        else:
            torch.save(self.state_dict(), path)

class ECNTrainer:
    def __init__(
        self,
        env,
        state_value_net=None,
        actor=None,
        critic=None,
        autoencoder=None,
        exploration_critic=None,
        gamma=args.gamma,
        seed=args.seed,
        print_logs=True,
        replay_buffer_size=args.buffer_size,
        actor_critic_timesteps=args.total_timesteps,
        polyak_constant=args.tau,
        policy_update_frequency=args.policy_frequency,
        learning_starts=args.learning_starts,
        exploration_noise=args.exploration_noise,
        actor_critic_batch_size=args.batch_size,
    ):
        self.env = env
        self.env.single_observation_space.dtype = np.float32
        self.seed = seed
        self.state_value_net = StateValueApproximator(env).to(device)
        self.print_logs = print_logs

        self.latent_size = np.array(env.single_observation_space.shape).prod() // 2

        self.autoencoder = StateAggregationAutoencoder(
            env, latent_size=self.latent_size
        )

        self.actor = Actor(env, latent_size=self.latent_size).to(device)
        self.critic = Critic(env, latent_size=self.latent_size).to(device)
        self.actor_target = Actor(env, latent_size=self.latent_size).to(device)
        self.critic_target = Critic(env, latent_size=self.latent_size).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.exploration_critic = ExplorationCritic(
            env, latent_state_size=self.latent_size
        )
        self.exploration_buffer = ExplorationBuffer()

        if autoencoder:
            self.autoencoder.load_state_dict(torch.load(autoencoder))

        if state_value_net:
            self.state_value_net.load_state_dict(torch.load(state_value_net))

        if actor:
            self.actor.load_state_dict(torch.load(actor))
            self.actor_target.load_state_dict(torch.load(actor))

        if critic:
            self.critic.load_state_dict(torch.load(critic))
            self.critic_target.load_state_dict(torch.load(critic))

        if exploration_critic:
            self.exploration_critic.load_state_dict(torch.load(exploration_critic))

        # Algorithm hyperparameters
        self.gamma = gamma

        # State value net training hyperparameters
        self.state_value_net_batch_size = 128
        self.state_value_net_baseline_rollout_episodes = 100
        self.state_value_net_baseline_num_rollout = 10
        self.state_value_net_baseline_epochs = 1000
        self.state_value_net_learning_rate = 3e-4

        # Autoencoder training hyperparameters
        self.autoencoder_batch_size = 128
        self.autoencoder_learning_rate = 1e-4
        self.autoencoder_epochs = 200
        self.autoencoder_num_samples = 10000
        self.autoencoder_regularization_coefficient = 1e-4
        self.autoencoder_validation_frequency = 10

        # Autoencoder Single Step Training Hyperparameters
        self.autoencoder_num_updates = 1

        # State value function single step training hyperparameters
        self.state_value_net_num_updates = 10

        # Actor training hyperparameters
        self.actor_learning_rate = 3e-4
        self.actor_batch_size = 128

        # Critic training hyperparameters
        self.critic_learning_rate = 3e-4
        self.critic_batch_size = 128

        self.replay_buffer = ReplayBuffer(
            replay_buffer_size,
            self.env.single_observation_space,
            self.env.single_action_space,
            device,
            handle_timeout_termination=False,
        )

        # Actor Critic Algorithm Hyperparameters
        self.actor_critic_timesteps = actor_critic_timesteps
        self.polyak_constant = polyak_constant
        self.policy_update_frequency = policy_update_frequency
        self.learning_starts = learning_starts
        self.exploration_noise = exploration_noise
        self.actor_critic_batch_size = actor_critic_batch_size

        # Counters for logging
        self.state_value_net_loss_counter = 0
        self.state_aggregation_autoencoder_loss_counter = 0
        
        # Exploration critic hyperparameters
        self.exploration_buffer_num_experiences = 50
        self.exploration_critic_learning_rate = 1e-3
        

    def generate_rollouts(self, num_episodes=10):
        all_states = []
        all_returns = []
        all_next_states = []
        all_episodic_returns = []

        obs, _ = self.env.reset(seed=self.seed)
        episode_states = []
        episode_rewards = []
        episode_next_states = []
        current_episode = 0
        current_step = 0
        while current_episode < num_episodes:
            if self.print_logs:
                print(
                    f"Generating Rollout: Episode: {current_episode + 1}/{num_episodes}, Step: {current_step+1}/{envs.envs[0].spec.max_episode_steps}"
                )
            done = np.array([False] * self.env.num_envs)

            actions = np.array(
                [
                    self.env.single_action_space.sample()
                    for _ in range(self.env.num_envs)
                ]
            )
            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            done = np.logical_or(terminations, truncations)
            episode_states.append(obs)
            episode_rewards.append(rewards)
            episode_next_states.append(next_obs)

            obs = next_obs
            current_step += 1
            for _, is_done in enumerate(done):
                if is_done:
                    episode_states = np.array(episode_states)
                    episode_rewards = np.array(episode_rewards)
                    episode_next_states = np.array(episode_next_states)

                    T = episode_rewards.shape[0]
                    discounted_returns = np.empty_like(episode_rewards)
                    discounted_returns[-1] = episode_rewards[-1]
                    for t in range(T - 2, -1, -1):
                        discounted_returns[t] = (
                            episode_rewards[t] + self.gamma * discounted_returns[t + 1]
                        )

                    all_states.append(
                        episode_states.reshape(-1, *episode_states.shape[2:])
                    )
                    all_next_states.append(
                        episode_next_states.reshape(-1, *episode_next_states.shape[2:])
                    )
                    all_returns.append(discounted_returns.reshape(-1))
                    all_episodic_returns.append(
                        np.array(
                            [infos["final_info"]["episode"]["r"][-1]]
                            * np.array(discounted_returns.shape).prod()
                        )
                    )

                    episode_states = []
                    episode_rewards = []
                    episode_next_states = []
                    current_episode += 1
                    current_step = 0
                    break

        all_states = np.concatenate(all_states, axis=0)
        all_next_states = np.concatenate(all_next_states, axis=0)
        all_returns = np.concatenate(all_returns, axis=0)
        all_episodic_returns = np.concatenate(all_episodic_returns, axis=0)
        return all_states, all_returns, all_next_states, all_episodic_returns

    def train_baseline_state_value_network(self):
        self.state_value_net.train()

        optimizer = optim.Adam(
            self.state_value_net.parameters(), lr=self.state_value_net_learning_rate
        )
        loss_fn = nn.MSELoss()
        for rollout in range(self.state_value_net_baseline_rollout_episodes):
            states_np, returns_np, _, _ = self.generate_rollouts(
                self.state_value_net_baseline_num_rollout
            )
            states_tensor = torch.tensor(states_np, device=device, dtype=torch.float32)
            returns_tensor = torch.tensor(
                returns_np, device=device, dtype=torch.float32
            )

            dataset = TensorDataset(states_tensor, returns_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=self.state_value_net_batch_size,
                shuffle=True,
                generator=torch.Generator(device=device),
            )

            for epoch in range(self.state_value_net_baseline_epochs):
                if self.print_logs:
                    print(
                        f"Training State Value Network - Rollout: {rollout + 1}/{self.state_value_net_baseline_num_rollout}, Epoch: {epoch + 1}/{self.state_value_net_baseline_epochs}"
                    )
                epoch_loss = 0.0
                for batch_states, batch_returns in dataloader:
                    optimizer.zero_grad()
                    predictions = self.state_value_net(batch_states)
                    loss = loss_fn(predictions, batch_returns)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * batch_states.size(0)
                avg_loss = epoch_loss / len(dataset)
                writer.add_scalar(
                    "Loss/state_value_network_loss",
                    avg_loss,
                    self.state_value_net_loss_counter,
                )
                self.state_value_net_loss_counter += 1

        self.state_value_net.save()
        self.state_value_net.eval()

    def train_baseline_state_aggregation_module(self):
        self.autoencoder.train()
        optimizer = optim.Adam(
            self.autoencoder.parameters(), lr=self.autoencoder_learning_rate
        )
        states_np, returns_np, next_states_np, episodic_returns = (
            self.generate_rollouts(self.state_value_net_baseline_num_rollout)
        )
        dataset = RolloutDataset(
            states_np, returns_np, next_states_np, episodic_returns
        )
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator(device=device),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.autoencoder_batch_size,
            shuffle=True,
            generator=torch.Generator(device=device),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.autoencoder_batch_size,
            shuffle=False,
            generator=torch.Generator(device=device),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.autoencoder_batch_size,
            shuffle=False,
            generator=torch.Generator(device=device),
        )

        for epoch in range(self.autoencoder_epochs):
            if self.print_logs:
                print(
                    f"Training State Aggregation Module - Epoch: {epoch + 1}/{self.autoencoder_epochs}"
                )

            for batch in train_loader:
                current_state, _, next_states, episodic_return = batch
                current_state = current_state.to(device)
                current_state.requires_grad = True

                encoder_out, current_state_pred, next_state_pred = self.autoencoder(
                    current_state
                )
                loss = self.autoencoder.compute_loss(
                    encoder_out,
                    next_state_pred,
                    next_states,
                    current_state_pred,
                    current_state,
                    episodic_return,
                    self.autoencoder_regularization_coefficient,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            writer.add_scalar(
                "Loss/state_aggregation_autoencoder_loss",
                loss,
                self.state_aggregation_autoencoder_loss_counter,
            )
            self.state_aggregation_autoencoder_loss_counter += 1

            if epoch % self.autoencoder_validation_frequency == 0:
                self.autoencoder.eval()
                val_loss = 0.0
                for batch in val_loader:
                    current_state, _, next_states, episodic_return = batch
                    current_state = current_state.to(device)
                    current_state.requires_grad = True
                    encoder_out, current_state_pred, next_state_pred = self.autoencoder(
                        current_state
                    )
                    loss = self.autoencoder.compute_loss(
                        encoder_out,
                        next_state_pred,
                        next_states,
                        current_state_pred,
                        current_state,
                        episodic_return,
                        self.autoencoder_regularization_coefficient,
                    )
                    val_loss += loss.item()

                val_loss /= len(val_loader)
                writer.add_scalar("Loss/validation_autoencoder_loss", val_loss, epoch)
                self.autoencoder.train()

        # Final test evaluation
        self.autoencoder.eval()
        test_loss = 0.0
        for batch in test_loader:
            current_state, _, next_states, episodic_return = batch
            current_state = current_state.to(device)
            current_state.requires_grad = True
            encoder_out, current_state_pred, next_state_pred = self.autoencoder(
                current_state
            )
            loss = self.autoencoder.compute_loss(
                encoder_out,
                next_state_pred,
                next_states,
                current_state_pred,
                current_state,
                episodic_return,
                self.autoencoder_regularization_coefficient,
            )
            test_loss += loss.item()
        test_loss /= len(test_loader)

        if self.print_logs:
            print(f"Final Test Loss: {test_loss}")

        self.state_value_net.train()
        self.autoencoder.save()
        self.autoencoder.eval()

    def train_single_step_state_aggregation_module(
        self, observations, next_observations, episodic_return
    ):
        next_states = torch.tensor(
            next_observations, requires_grad=True, dtype=torch.float32
        )
        self.autoencoder.train()
        optimizer = torch.optim.Adam(
            lr=self.autoencoder_learning_rate, params=self.autoencoder.parameters()
        )

        current_state = torch.tensor(
            observations, requires_grad=True, dtype=torch.float32
        )

        for i in range(self.autoencoder_num_updates):
            encoder_out, current_state_pred, next_state_pred = self.autoencoder(
                current_state
            )
            loss = self.autoencoder.compute_loss(
                encoder_out,
                next_state_pred,
                next_states,
                current_state_pred,
                current_state,
                episodic_return,
                self.autoencoder_regularization_coefficient,
            )
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            writer.add_scalar(
                "Loss/state_aggregation_autoencoder_loss",
                loss,
                self.state_aggregation_autoencoder_loss_counter,
            )
            self.state_aggregation_autoencoder_loss_counter += 1

        self.autoencoder.eval()

    def train_single_step_state_value_network(self, episode_states, episode_rewards):
        self.state_value_net.train()
        episode_states = np.array(episode_states)
        episode_rewards = np.array(episode_rewards)

        T = episode_rewards.shape[0]
        discounted_returns = np.empty_like(episode_rewards)
        discounted_returns[-1] = episode_rewards[-1]
        for t in range(T - 2, -1, -1):
            discounted_returns[t] = (
                episode_rewards[t] + self.gamma * discounted_returns[t + 1]
            )

        discounted_returns = torch.tensor(
            discounted_returns.reshape(-1), dtype=torch.float32
        ).unsqueeze(-1)
        optimizer = optim.Adam(
            self.state_value_net.parameters(), lr=self.state_value_net_learning_rate
        )

        for i in range(self.state_value_net_num_updates):
            prediction = self.state_value_net(
                torch.tensor(episode_states, dtype=torch.float32)
            )
            loss = F.mse_loss(prediction, discounted_returns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar(
                "Loss/state_value_network_loss",
                loss,
                self.state_value_net_loss_counter,
            )
            self.state_value_net_loss_counter += 1

        self.state_value_net.eval()

    def train_single_step_exploration_critic(self, latent_state, action, latent_next_state):
        optimizer = optim.Adam(
            list(self.exploration_critic.parameters()), lr=self.critic_learning_rate
        )
        recent_experiences = self.exploration_buffer.get_last_k_elements(self.exploration_buffer_num_experiences)
        if len(recent_experiences) < self.exploration_buffer_num_experiences:
            if self.print_logs:
                print(f"Training Exploration Critic: Insufficient experience: Current - {len(recent_experiences)}, Need - {self.exploration_buffer_num_experiences}")
            return
        
        buffer_seq_list = []
        for exp in recent_experiences:
            concatenated = torch.cat([exp.latent_state, exp.action, exp.latent_next_state], dim=-1)
            buffer_seq_list.append(concatenated)
        
        exploration_buffer_seq = torch.stack(buffer_seq_list)
        exploration_score = self.exploration_critic(latent_state, action, latent_next_state, exploration_buffer_seq)
        distance = torch.linalg.norm(latent_next_state - latent_state)
        loss = F.mse_loss(exploration_score, distance)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    def train(self):
        self.state_value_net.eval()
        self.autoencoder.eval()

        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

        critic_optimizer = optim.Adam(
            list(self.critic.parameters()), lr=self.critic_learning_rate
        )
        actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=self.actor_learning_rate
        )

        obs, _ = self.env.reset(seed=self.seed)
        current_episode_states = []
        current_episode_rewards = []
        current_episode_next_states = []
        for global_step in range(self.actor_critic_timesteps):
            encoded_obs = self.autoencoder.encoder(torch.tensor(obs, dtype=torch.float32).to(device))
            if global_step < self.learning_starts:
                actions = np.array(
                    [envs.single_action_space.sample() for _ in range(envs.num_envs)]
                )
            else:
                with torch.no_grad():
                    actions = self.actor(
                        encoded_obs
                    )
                    actions += torch.normal(
                        0, self.actor.action_scale * self.exploration_noise
                    )
                    actions = (
                        actions.cpu()
                        .numpy()
                        .clip(
                            self.env.single_action_space.low, self.env.single_action_space.high
                        )
                    )

            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            
            encoded_next_obs = self.autoencoder.encoder(torch.tensor(next_obs, dtype=torch.float32))
            self.exploration_buffer.add(encoded_obs, actions.squeeze(0), encoded_next_obs)
            current_episode_states.append(obs.squeeze(0))
            current_episode_rewards.append(rewards.squeeze(0))
            current_episode_next_states.append(next_obs.squeeze(0))
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for reward in infos["final_info"]["episode"]["r"]:
                    if self.print_logs:
                        print(f"global_step={global_step}, episodic_return={reward}")
                    writer.add_scalar("charts/episodic_return", reward, global_step)
                    break

                for length in infos["final_info"]["episode"]["l"]:
                    writer.add_scalar("charts/episodic_length", length, global_step)
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, is_done in enumerate(np.logical_or(terminations, truncations)):
                if is_done:
                    real_next_obs[idx] = infos["final_obs"][idx]
                    self.train_single_step_state_aggregation_module(
                        current_episode_states, current_episode_next_states, infos["final_info"]["episode"]["r"][-1]
                    )
                    current_episode_states = []
                    current_episode_rewards = []
                    current_episode_next_states = []

            self.replay_buffer.add(
                obs, real_next_obs, actions, rewards, terminations, infos
            )

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            self.train_single_step_exploration_critic(encoded_obs, actions.squeeze(0), encoded_next_obs)
            if global_step > self.learning_starts:
                data = self.replay_buffer.sample(self.actor_critic_batch_size)
                encoded_data_obs = self.autoencoder.encoder(data.observations)
                encoded_data_next_obs = self.autoencoder.encoder(data.next_observations)
                
                with torch.no_grad():
                    next_state_actions = self.actor_target(encoded_data_next_obs)
                    qf1_next_target = self.critic_target(data.next_observations, next_state_actions)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)
                
                qf1_a_values = self.critic(
                    data.observations,
                    data.actions,
                ).view(-1)

                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

                # optimize the model
                critic_optimizer.zero_grad()
                qf1_loss.backward()
                critic_optimizer.step()

                if global_step % self.policy_update_frequency == 0:
                    actor_action = self.actor(encoded_data_obs)                    
                    actor_loss = -self.critic(
                        data.observations, actor_action,
                    ).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    if self.print_logs:
                        print(
                            f"Training Actor Critic: Current timestep: {global_step}/{self.actor_critic_timesteps}, Actor Loss: {actor_loss}, Critic Loss: {qf1_loss}"
                        )

                    # update the target network
                    for param, target_param in zip(
                        self.actor.parameters(), self.actor_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.polyak_constant * param.data
                            + (1 - self.polyak_constant) * target_param.data
                        )
                    for param, target_param in zip(
                        self.critic.parameters(), self.critic_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.polyak_constant * param.data
                            + (1 - self.polyak_constant) * target_param.data
                        )
                
                if global_step % 100 == 0:
                    writer.add_scalar(
                        "Loss/qf1_values", qf1_a_values.mean().item(), global_step
                    )
                    writer.add_scalar("Loss/critic_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("Loss/actor_loss", actor_loss.item(), global_step)


        self.env.close()
        self.actor.save()
        self.critic.save()
        self.state_value_net.train()
        self.autoencoder.train()


if __name__ == "__main__":
    ecn_trainer = ECNTrainer(
        envs,
        print_logs=False
        # state_value_net="./training_checkpoints/state_value_approximator.pt",
    )
    # ecn_trainer.train_baseline_state_value_network()
    # ecn_trainer.train_baseline_state_aggregation_module()
    ecn_trainer.train()
