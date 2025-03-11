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
    env_id: str = "BipedalWalker-v3"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-3
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 10
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""


args = tyro.cli(Args)
run_name = f"runs/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
os.makedirs(run_name, exist_ok=True)
os.makedirs(f"{run_name}/models", exist_ok=True)
os.makedirs(f"{run_name}/logs", exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
torch.set_default_device(device)

writer = SummaryWriter(log_dir=f"{run_name}/logs")


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name.replace('runs/', '')}"
            )
            env.recorded_frames = []
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
)

ExplorationBufferElement = namedtuple(
    "ExplorationBufferElement", "latent_state action latent_next_state"
)

class StateValueApproximator(nn.Module):  
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod(),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm1d(256)
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

    def save(self):
        torch.save(self, f"{run_name}/models/state_value_approximator.pt")

# ALGO LOGIC: initialize agent here:
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
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def save(self):
        torch.save(self, f"{run_name}/models/critic.pt")


class StateAggregationEncoder(nn.Module):
    def __init__(self, env, latent_size=64):
        super(StateAggregationEncoder, self).__init__()
        self.latent_size = latent_size

        # Input Shape is flattened state space + Q Value
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + 1, 256
        )
        self.batch_norm_1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm_2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, self.latent_size)
        self.batch_norm_3 = nn.BatchNorm1d(self.latent_size)

    def forward(self, state, value):
        x = self.fc1(torch.tensor(torch.hstack((state, value)), device=device))
        x = self.batch_norm_1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.batch_norm_3(x)

        return F.softmax(x).to(device)

    def save(self):
        torch.save(self, f"{run_name}/models/state_aggregation_encoder.pt")


class StateAggregationDecoder(nn.Module):
    def __init__(self, env, latent_size=64):
        super(StateAggregationDecoder, self).__init__()

        self.latent_size = latent_size

        self.fc1 = nn.Linear(self.latent_size, 128)
        self.batch_norm_1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.batch_norm_2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, np.array(env.single_observation_space.shape).prod())

    def forward(self, latent_state):
        x = self.fc1(latent_state)
        x = self.batch_norm_1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x

    def save(self):
        torch.save(self, f"{run_name}/models/state_aggregation_decoder.pt")


class StateAggregationAutoencoder(nn.Module):
    def __init__(self, env: gym.Env, latent_size=64):
        super(StateAggregationAutoencoder, self).__init__()

        self.encoder = StateAggregationEncoder(env, latent_size=latent_size)
        self.decoder = StateAggregationDecoder(env, latent_size=latent_size)

    def forward(self, state, value):
        return self.decoder(torch.tensor(self.encoder(state, value), device=device))

    def save(self):
        self.encoder.save()
        self.decoder.save()
        torch.save(self, f"{run_name}/models/state_aggregation_autoencoder.pt")


# Actor by default works in latent space
class Actor(nn.Module):
    def __init__(self, env, latent_size=64):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc_mu = nn.Linear(256, np.array(env.single_action_space.shape).prod())

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
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        mu = self.fc_mu(x)
        return mu

    def save(self):
        torch.save(self, f"{run_name}/models/actor.pt")


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, length, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        positional_encoding = torch.zeros(length, embedding_size)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
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
        torch.save(self, f"{run_name}/models/exploration_critic.pt")


def train_state_value_network():
    NUM_EPISODES = 100
    BATCH_SIZE = 128
    EPOCHS = 100

    def generate_rollouts(num_episodes=NUM_EPISODES, gamma=args.gamma):
        all_states = []
        all_returns = []
        for ep in range(num_episodes):
            obs, _ = envs.reset(seed=args.seed + ep)
            done = np.array([False] * envs.num_envs)
            episode_states = []
            episode_rewards = []

            while (not done.all()):
                episode_states.append(obs)
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)
                episode_rewards.append(rewards)
                done = np.logical_or(terminations, truncations)
                obs = next_obs

            episode_states = np.array(episode_states)
            episode_rewards = np.array(episode_rewards)
            T = episode_rewards.shape[0]

            discounted_returns = np.empty_like(episode_rewards)
            discounted_returns[-1] = episode_rewards[-1]
            for t in range(T - 2, -1, -1):
                discounted_returns[t] = episode_rewards[t] + gamma * discounted_returns[t + 1]

            num_envs = envs.num_envs
            all_states.append(episode_states.reshape(-1, *episode_states.shape[2:]))
            all_returns.append(discounted_returns.reshape(-1))

        all_states = np.concatenate(all_states, axis=0)
        all_returns = np.concatenate(all_returns, axis=0).reshape(-1, 1)
        return all_states, all_returns

    states_np, returns_np = generate_rollouts()
    states_tensor = torch.tensor(states_np, dtype=torch.float32, device=device)
    returns_tensor = torch.tensor(returns_np, dtype=torch.float32, device=device)

    dataset = TensorDataset(states_tensor, returns_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    state_value_net = StateValueApproximator(envs).to(device)
    optimizer = optim.Adam(state_value_net.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_states, batch_returns in dataloader:
            optimizer.zero_grad()
            predictions = state_value_net(batch_states)
            loss = loss_fn(predictions, batch_returns)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_states.size(0)
        avg_loss = epoch_loss / len(dataset)
        writer.add_scalar("Loss/state_value_network_loss", avg_loss, epoch)
    
    state_value_net.save()
    return state_value_net

def train_actor_critic(actor=None, critic=None, total_timesteps=args.total_timesteps):

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    if not actor:
        actor = Actor(envs).to(device)
        target_actor = Actor(envs).to(device)
        target_actor.load_state_dict(actor.state_dict())

    if not critic:
        qf1 = Critic(envs).to(device)
        qf1_target = Critic(envs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            with torch.no_grad():
                actor.eval()
                actions = actor(torch.Tensor(obs).to(device))
                actor.train()
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = (
                    actions.cpu()
                    .numpy()
                    .clip(envs.single_action_space.low, envs.single_action_space.high)
                )

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        rb.add(obs, next_obs, actions, rewards, terminations, infos)

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                target_actor.eval()
                next_state_actions = target_actor(data.next_observations)
                target_actor.train()
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            writer.add_scalar("Loss/critic_loss", qf1_loss, global_step)
            writer.add_scalar("Loss/qf1_values", qf1_a_values.mean(), global_step)

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                writer.add_scalar("Loss/actor_loss", actor_loss, global_step)

                # update the target network
                for param, target_param in zip(
                    actor.parameters(), target_actor.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

    envs.close()
    target_actor.save()
    qf1_target.save()
    return target_actor, qf1_target


def train_state_aggregation_module(state_value_net):
    autoencoder = StateAggregationAutoencoder(envs)
    BATCH_SIZE = 64
    SAMPLED_ACTION_COUNT = 1000
    NUM_SAMPLES = 10000
    LEANRING_RATE = 1e-2
    EPOCHS = 101
    optimizer = torch.optim.Adam(lr=LEANRING_RATE, params=autoencoder.parameters())

    dataset = [envs.single_observation_space.sample() for i in range(NUM_SAMPLES)]
    loss_fn = nn.MSELoss()

    for i in range(EPOCHS):
        sample = torch.tensor(
            random.sample(dataset, BATCH_SIZE), dtype=torch.float32, device=device
        )
        value = approximate_value(sample)
        autoencoder_out = autoencoder(sample, value)
        loss = loss_fn(autoencoder_out, sample)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/state_aggregation_autoencoder_loss", loss, i)

    return autoencoder.encoder, autoencoder.decoder, autoencoder


def train_exploration_critic(actor, state_aggregation_encoder):
    pass


if __name__ == "__main__":
    state_value_network = train_state_value_network()
    state_aggregation_encoder, _, _ = train_state_aggregation_module(state_value_network)

    actor, critic = train_actor_critic()
    
    pipeline_loops = 10
    for i in range(pipeline_loops):
        # state_aggregation_encoder, _, _ = train_state_aggregation_module(actor, critic)
        # exploration_critic = train_exploration_critic(actor, state_aggregation_encoder)
        # actor, critic = train_actor_critic(actor, critic, exploration_critic)
        pass
