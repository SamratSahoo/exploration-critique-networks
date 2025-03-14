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
from torch.utils.data import TensorDataset, DataLoader
import stable_baselines3 as sb3

os.environ["MUJOCO_GL"] = "osmesa"


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
    total_timesteps: int = 1000000
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


def contractive_loss(encoder_out, decoder_out, data, lmbda):
    criterion = nn.MSELoss()
    mse_loss = criterion(decoder_out, data)

    encoder_out.backward(torch.ones(encoder_out.size()), retain_graph=True)
    loss2 = torch.sqrt(torch.sum(torch.pow(data.grad, 2)))
    data.grad.data.zero_()
    loss = mse_loss + (lmbda * loss2)
    return loss


envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)],
    autoreset_mode=gym.vector.vector_env.AutoresetMode.SAME_STEP,
)

ExplorationBufferElement = namedtuple(
    "ExplorationBufferElement", "latent_state action latent_next_state"
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
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)

        self.obs_space_size = np.array(env.single_observation_space.shape).prod()

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        # x = self.bn2(x)
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
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + 1, 256
        )
        self.batch_norm_1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm_2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, self.latent_size)
        self.batch_norm_3 = nn.BatchNorm1d(self.latent_size)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

    def forward(self, state, value):
        x = self.fc1(torch.cat((state, value), dim=1))
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

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

    def forward(self, latent_state):
        x = self.fc1(latent_state)
        x = self.batch_norm_1(x)
        x = self.prelu1(x)

        x = self.fc2(x)
        x = self.batch_norm_2(x)
        x = self.prelu2(x)

        x = self.fc3(x)
        return x

    def save(self, path=None):
        if path:
            torch.save(self.state_dict(), path)
        else:
            torch.save(
                self.state_dict(), f"{run_name}/models/state_aggregation_decoder.pt"
            )


class StateAggregationAutoencoder(nn.Module):
    def __init__(self, env: gym.Env, latent_size=64):
        super(StateAggregationAutoencoder, self).__init__()

        self.encoder = StateAggregationEncoder(env, latent_size=latent_size)
        self.decoder = StateAggregationDecoder(env, latent_size=latent_size)

    def forward(self, state, value):
        encoder_out = self.encoder(state, value)
        decoder_out = self.decoder(encoder_out)
        return encoder_out, decoder_out

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
            latent_state_size + np.array(env.single_action_space.shape).prod(),
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

    def forward(self, latent_state, latent_next_state, action, exploration_buffer_seq):
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

    def save(self):
        torch.save(self.state_dict(), f"{run_name}/models/exploration_critic.pt")


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
        self.seed = seed
        self.state_value_net = StateValueApproximator(env).to(device)
        self.print_logs = print_logs

        self.latent_size = np.array(env.single_observation_space.shape).prod() // 2

        self.autoencoder = StateAggregationAutoencoder(
            env, latent_size=self.latent_size
        )

        self.actor = Actor(env, latent_size=self.latent_size)
        self.critic = Critic(env)
        self.actor_target = Actor(env, latent_size=self.latent_size)
        self.critic_target = Critic(env)

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
        self.autoencoder_learning_rate = 1e-3
        self.autoencoder_epochs = 200
        self.autoencoder_num_samples = 10000
        self.autoencoder_regularization_coefficient = 1e-4
        self.autoencoder_validation_frequency = 10

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

    def train_baseline_state_value_network(self):
        self.state_value_net.train()

        def generate_rollouts():
            all_states = []
            all_returns = []

            obs, _ = self.env.reset(seed=self.seed)
            episode_states = []
            episode_rewards = []
            current_episode = 0
            current_step = 0
            while current_episode < self.state_value_net_baseline_rollout_episodes:
                if self.print_logs:
                    print(
                        f"Generating Rollout: Episode: {current_episode + 1}/{self.state_value_net_baseline_rollout_episodes}, Step: {current_step+1}/{envs.envs[0].spec.max_episode_steps}"
                    )
                done = np.array([False] * self.env.num_envs)

                actions = np.array(
                    [
                        self.env.single_action_space.sample()
                        for _ in range(self.env.num_envs)
                    ]
                )
                next_obs, rewards, terminations, truncations, _ = self.env.step(actions)
                done = np.logical_or(terminations, truncations)
                episode_states.append(obs)
                episode_rewards.append(rewards)

                obs = next_obs
                current_step += 1
                for _, is_done in enumerate(done):
                    if is_done:
                        episode_states = np.array(episode_states)
                        episode_rewards = np.array(episode_rewards)

                        T = episode_rewards.shape[0]
                        discounted_returns = np.empty_like(episode_rewards)
                        discounted_returns[-1] = episode_rewards[-1]
                        for t in range(T - 2, -1, -1):
                            discounted_returns[t] = (
                                episode_rewards[t]
                                + self.gamma * discounted_returns[t + 1]
                            )

                        all_states.append(
                            episode_states.reshape(-1, *episode_states.shape[2:])
                        )
                        all_returns.append(discounted_returns.reshape(-1))

                        episode_states = []
                        episode_rewards = []
                        current_episode += 1
                        current_step = 0
                        break

            all_states = np.concatenate(all_states, axis=0)
            all_returns = np.concatenate(all_returns, axis=0).reshape(-1, 1)
            return all_states, all_returns

        optimizer = optim.Adam(
            self.state_value_net.parameters(), lr=self.state_value_net_learning_rate
        )
        loss_fn = nn.MSELoss()
        for rollout in range(self.state_value_net_baseline_num_rollout):
            states_np, returns_np = generate_rollouts()
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
                    rollout * self.state_value_net_baseline_epochs + epoch,
                )

        self.state_value_net.save()
        self.state_value_net.eval()

    def train_baseline_state_aggregation_module(self):
        self.autoencoder.train()
        optimizer = optim.Adam(
            self.autoencoder.parameters(), lr=self.autoencoder_learning_rate
        )

        dataset = torch.tensor(
            [
                envs.single_observation_space.sample()
                for _ in range(self.autoencoder_num_samples)
            ],
            dtype=torch.float32,
            device=device,
        )

        # Split dataset into train, validation, and test sets
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
                batch = batch.to(device)
                batch.requires_grad = True
                value = self.state_value_net(batch)
                encoder_out, decoder_out = self.autoencoder(batch, value)
                loss = contractive_loss(
                    encoder_out,
                    decoder_out,
                    batch,
                    self.autoencoder_regularization_coefficient,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            writer.add_scalar("Loss/state_aggregation_autoencoder_loss", loss, epoch)

            if epoch % self.autoencoder_validation_frequency == 0:
                self.autoencoder.eval()
                val_loss = 0.0
                for batch in val_loader:
                    batch = batch.to(device)
                    batch.requires_grad = True
                    value = self.state_value_net(batch)
                    encoder_out, decoder_out = self.autoencoder(batch, value)
                    val_loss += contractive_loss(
                        encoder_out,
                        decoder_out,
                        batch,
                        self.autoencoder_regularization_coefficient,
                    ).item()
                val_loss /= len(val_loader)
                writer.add_scalar("Loss/validation_autoencoder_loss", val_loss, epoch)
                self.autoencoder.train()

        # Final test evaluation
        self.autoencoder.eval()
        test_loss = 0.0
        for batch in test_loader:
            batch = batch.to(device)
            batch.requires_grad = True
            value = self.state_value_net(batch)
            encoder_out, decoder_out = self.autoencoder(batch, value)
            test_loss += contractive_loss(
                encoder_out,
                decoder_out,
                batch,
                self.autoencoder_regularization_coefficient,
            ).item()
        test_loss /= len(test_loader)

        if self.print_logs:
            print(f"Final Test Loss: {test_loss}")

        self.state_value_net.train()
        self.autoencoder.save()
        self.autoencoder.eval()

    def train_single_step_state_aggregation_module(self, observations):
        self.autoencoder.train()
        optimizer = torch.optim.Adam(
            lr=self.autoencoder_learning_rate, params=self.autoencoder.parameters()
        )

        observations = torch.tensor(observations, requires_grad=True)
        value = self.state_value_net(observations)
        encoder_out, decoder_out = self.autoencoder(observations, value)
        loss = contractive_loss(
            encoder_out,
            decoder_out,
            observations,
            self.autoencoder_regularization_coefficient,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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

        episodes_states = episode_states.reshape(
            -1, *episode_states.shape[2:]
        ).unsqueeze(0)
        discounted_returns = discounted_returns.reshape(-1).unsqueeze(0)

        optimizer = optim.Adam(
            self.state_value_net.parameters(), lr=self.state_value_net_learning_rate
        )

        prediction = self.state_value_net(episodes_states)
        loss = F.mse_loss(prediction, discounted_returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.state_value_net.eval()

    def train_single_step_exploration_critic(self):
        pass

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
        value = self.state_value_net(torch.tensor(obs, dtype=torch.float32))
        encoded_obs = self.autoencoder.encoder(
            torch.tensor(obs, dtype=torch.float32), value
        )
        current_episode_states = []
        current_episode_rewards = []
        for global_step in range(self.actor_critic_timesteps):
            current_episode_states.append(obs)
            if global_step < self.learning_starts:
                actions = np.array(
                    [envs.single_action_space.sample() for _ in range(envs.num_envs)]
                )
            else:
                with torch.no_grad():
                    self.actor.eval()
                    actions = self.actor(
                        torch.tensor(encoded_obs, dtype=torch.float32).to(device)
                    )
                    self.actor.train()
                    actions += torch.normal(
                        0, self.actor.action_scale * self.exploration_noise
                    )
                    actions = (
                        actions.cpu()
                        .numpy()
                        .clip(
                            envs.single_action_space.low, envs.single_action_space.high
                        )
                    )

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            current_episode_rewards.append(rewards)
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for reward in infos["final_info"]["episode"]["r"]:
                    print(f"global_step={global_step}, episodic_return={reward}")
                    writer.add_scalar("charts/episodic_return", reward, global_step)
                    break

                for length in infos["final_info"]["episode"]["l"]:
                    writer.add_scalar("charts/episodic_length", length, global_step)
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(np.logical_or(terminations, truncations)):
                if trunc:
                    real_next_obs[idx] = infos["final_obs"][idx]
                    self.train_single_step_state_aggregation_module(
                        current_episode_states
                    )
                    self.train_single_step_state_value_network(
                        current_episode_states, current_episode_rewards
                    )
                    current_episode_states = []

            self.replay_buffer.add(
                obs, real_next_obs, actions, rewards, terminations, infos
            )

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            if global_step > self.learning_starts:
                data = self.replay_buffer.sample(self.actor_critic_batch_size)
                with torch.no_grad():
                    self.target_actor.eval()

                    next_values = self.state_value_net(
                        torch.tensor(data.next_observations, dtype=torch.float32)
                    )
                    encoded_next_obs = self.autoencoder.encoder(
                        torch.tensor(data.next_observations, dtype=torch.float32),
                        next_values,
                    )
                    next_state_actions = self.target_actor(
                        torch.tensor(encoded_next_obs, dtype=torch.float32)
                    )

                    self.target_actor.train()
                    qf1_next_target = self.critic_target(
                        torch.tensor(data.next_observations, dtype=torch.float32),
                        torch.tensor(next_state_actions, dtype=torch.float32),
                    )
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.gamma * (qf1_next_target).view(-1)

                qf1_a_values = self.critic(
                    torch.tensor(data.observations, dtype=torch.float32),
                    torch.tensor(data.actions, dtype=torch.float32),
                ).view(-1)

                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

                # optimize the model
                critic_optimizer.zero_grad()
                qf1_loss.backward()
                critic_optimizer.step()

                writer.add_scalar("Loss/critic_loss", qf1_loss, global_step)
                writer.add_scalar("Loss/qf1_values", qf1_a_values.mean(), global_step)

                if global_step % self.policy_update_frequency == 0:

                    next_values = self.state_value_net(
                        torch.tensor(data.next_observations, dtype=torch.float32)
                    )
                    encoded_next_obs = self.autoencoder.encoder(
                        torch.tensor(data.next_observations, dtype=torch.float32),
                        next_values,
                    )

                    actor_loss = -self.critic(
                        torch.tensor(data.observations, dtype=torch.float32),
                        torch.tensor(self.actor(encoded_next_obs), dtype=torch.float32),
                    ).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    if self.print_logs:
                        print(
                            f"Training Actor Critic: Current timestep: {global_step}/{self.actor_critic_timesteps}, Actor Loss: {actor_loss}, Critic Loss: {qf1_loss}"
                        )

                    writer.add_scalar("Loss/actor_loss", actor_loss, global_step)

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

        self.env.close()
        self.actor.save()
        self.critic.save()
        self.state_value_net.train()
        self.autoencoder.train()


if __name__ == "__main__":
    ecn_trainer = ECNTrainer(
        envs, state_value_net="./training_checkpoints/state_value_approximator.pt"
    )
    # ecn_trainer.train_baseline_state_value_network()
    ecn_trainer.train_baseline_state_aggregation_module()
    # state_value_network = train_baseline_state_value_network()
    # state_value_network = "./training_checkpoints/state_value_approximator.pt"
    # state_aggregation_encoder = "./training_checkpoints/state_aggregation_encoder.pt"
    # state_aggregation_encoder, _, _ = train_baseline_state_aggregation_module(state_value_network)
    # actor, critic = train_actor_critic(state_aggregation_encoder, state_value_network)

    # pipeline_loops = 10
    # for i in range(pipeline_loops):
    #     # state_aggregation_encoder, _, _ = train_state_aggregation_module(actor, critic)
    #     # exploration_critic = train_exploration_critic(actor, state_aggregation_encoder)
    #     # actor, critic = train_actor_critic(actor, critic, exploration_critic)
    #     pass
