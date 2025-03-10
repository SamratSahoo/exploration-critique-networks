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

import stable_baselines3 as sb3

if sb3.__version__ < "2.0":
    raise ValueError(
        """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
    )


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
    print_log_frequency: int = 100
    """the frequency of printed_logs"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""


args = tyro.cli(Args)
run_name = f"runs/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
os.makedirs(run_name, exist_ok=True)
os.makedirs(f"{run_name}/models", exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
torch.set_default_device(device)


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


# ALGO LOGIC: initialize agent here:
class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.obs_space_size = np.array(env.single_observation_space.shape).prod()
        self.fc1 = nn.Linear(
            self.obs_space_size + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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


class StateAggregationAutoEncoder(nn.Module):
    def __init__(self, env: gym.Env, latent_size=64):
        super(StateAggregationAutoEncoder, self).__init__()

        self.encoder = StateAggregationEncoder(env, latent_size=latent_size)
        self.decoder = StateAggregationDecoder(env, latent_size=latent_size)

    def forward(self, state, value):
        return self.decoder(torch.tensor(self.encoder(state, value), device=device))

    def save(self):
        self.encoder.save()
        self.decoder.save()
        torch.save(self, f"{run_name}/models/state_aggregation_autoencoder.pt")


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.array(env.single_action_space.shape).prod())
        self.fc_std = nn.Linear(256, np.array(env.single_action_space.shape).prod())

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
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) + 1e-6
        return mu, std

    def save(self):
        torch.save(self, f"{run_name}/models/actor.pt")


class CompressedActor(nn.Module):
    def __init__(self, env, latent_size=64):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
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

    def save(self):
        torch.save(self, f"{run_name}/compressed_actor.pt")


class PositionalEncoding(nn.Module):
    pass


ExplorationBufferElement = namedtuple("ExplorationBufferElement", "latent_state action")


class ExplorationBuffer:

    def __init__(self, maxlen=5000):
        self.buffer = deque(maxlen=maxlen)

    def add(self, latent_state, action):
        self.buffer.append(ExplorationBufferElement(latent_state, action))


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
            transformer_embedding_size, length=max_seq_len
        )

        # Encoding layer for exploration buffer
        encode_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embedding_size, nhead=heads
        )
        self.exploration_transfromer = nn.TransformerEncoder(encode_layer, num_layers=8)

    def forward(self, latent_state, latent_next_state, action):
        pass

    def save(self):
        torch.save(self, f"{run_name}/models/exploration_critic.pt")


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
                actions = actor(torch.Tensor(obs).to(device))
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
                next_state_actions = target_actor(data.next_observations)
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

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

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

            if global_step % args.print_log_frequency == 0:
                print(f"Critic Loss: {qf1_loss}, Actor Loss: {actor_loss}")

    envs.close()
    target_actor.save()
    qf1_target.save()
    return target_actor, qf1_target


def train_state_aggregation_module(actor, critic):
    autoencoder = StateAggregationAutoEncoder(envs)
    BATCH_SIZE = 64
    SAMPLED_ACTION_COUNT = 1000
    NUM_SAMPLES = 10000
    LEANRING_RATE = 1e-2
    EPOCHS = 100
    optimizer = torch.optim.Adam(lr=LEANRING_RATE, params=autoencoder.parameters())

    def approximate_value(state):
        if isinstance(actor, CompressedActor):
            state = autoencoder.encoder(state)

        action_mean, action_std = actor(state)
        dist = torch.distributions.Normal(action_mean, action_std)

        sampled_actions = dist.sample((SAMPLED_ACTION_COUNT,))
        state = state.unsqueeze(0).expand(SAMPLED_ACTION_COUNT, -1, -1)
        state = state.reshape(-1, critic.obs_space_size)

        sampled_actions = sampled_actions.reshape(-1, 4)

        q_values = critic(state, sampled_actions)
        q_values = q_values.view(SAMPLED_ACTION_COUNT, 64, -1).mean(dim=0)

        return torch.tensor(q_values, device=device)

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

        if i % args.print_log_frequency == 0:
            print(f"State Aggregation Autoencoder Loss: {loss}")

    return autoencoder.encoder, autoencoder.decoder, autoencoder


#
def train_exploration_critic(actor, state_aggregation_encoder):
    pass


if __name__ == "__main__":
    actor, critic = train_actor_critic()

    pipeline_loops = 10
    for i in range(pipeline_loops):
        state_aggregation_encoder, _, _ = train_state_aggregation_module(actor, critic)
        # exploration_critic = train_exploration_critic(actor, state_aggregation_encoder)
        # actor, critic = train_actor_critic(actor, critic, exploration_critic)
