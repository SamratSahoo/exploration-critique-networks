# Base DDPG Implementation from CleanRL: https://github.com/vwxyzjn/cleanrl
import os
import random
import time
from dataclasses import dataclass
import gc

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sys import platform
from ddpg_eval import evaluate_ecn

# Models
from ecn.models.critic import Critic
from ecn.models.state_aggregation_autoencoder import StateAggregationAutoencoder
from ecn.models.actor import Actor
from ecn.models.exploration_critic import ExplorationCritic
from ecn.utils import ExplorationBuffer, RolloutDataset, TrajectoryReplayBuffer

os.environ["MUJOCO_GL"] = "glfw" if platform == "darwin" else "osmesa"
torch.autograd.set_detect_anomaly(True)

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

class ECNTrainer:
    def __init__(
        self,
        env,
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
        
        # Exploration critic hyperparameters
        self.exploration_buffer_num_experiences = 256
        self.exploration_critic_batch_size = 256
        self.exploration_critic_learning_rate = 1e-3
        self.exploration_score_decay = 0.99

        self.exploration_critic = ExplorationCritic(
            env, latent_state_size=self.latent_size, 
            max_seq_len=self.exploration_buffer_num_experiences
        )
        self.exploration_buffer = ExplorationBuffer()

        if autoencoder:
            self.autoencoder.load_state_dict(torch.load(autoencoder))

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

        self.state_aggregation_baseline_num_rollout = 10

        # Autoencoder training hyperparameters
        self.autoencoder_batch_size = 128
        self.autoencoder_learning_rate = 1e-4
        self.autoencoder_epochs = 200
        self.autoencoder_num_samples = 10000
        self.autoencoder_regularization_coefficient = 1e-4
        self.autoencoder_validation_frequency = 10
        self.autoencoder_batch_size = 128

        # Autoencoder Single Step Training Hyperparameters
        self.autoencoder_num_updates = 1


        # Actor training hyperparameters
        self.actor_learning_rate = 3e-4
        self.actor_batch_size = 128

        # Critic training hyperparameters
        self.critic_learning_rate = 3e-4
        self.critic_batch_size = 128

        self.replay_buffer = TrajectoryReplayBuffer(
            replay_buffer_size,
            self.env.single_observation_space,
            self.env.single_action_space,
            device,
            latent_size=self.latent_size,
            trajectory_length=self.exploration_buffer_num_experiences,
            handle_timeout_termination=False,
        )
        self.exploration_critic_update_frequency = 1000

        # Actor Critic Algorithm Hyperparameters
        self.actor_critic_timesteps = actor_critic_timesteps
        self.polyak_constant = polyak_constant
        self.policy_update_frequency = policy_update_frequency
        self.learning_starts = learning_starts
        self.exploration_noise = exploration_noise
        self.actor_critic_batch_size = actor_critic_batch_size

        # Counters for logging
        self.state_aggregation_autoencoder_loss_counter = 0

        self.global_step = 0

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
                    episode_states_np = np.array(episode_states)
                    episode_rewards_np = np.array(episode_rewards)
                    episode_next_states_np = np.array(episode_next_states)

                    T = episode_rewards_np.shape[0]
                    discounted_returns = np.empty_like(episode_rewards_np)
                    discounted_returns[-1] = episode_rewards_np[-1]
                    for t in range(T - 2, -1, -1):
                        discounted_returns[t] = (
                            episode_rewards_np[t] + self.gamma * discounted_returns[t + 1]
                        )

                    all_states.append(
                        episode_states_np.reshape(-1, *episode_states_np.shape[2:])
                    )
                    all_next_states.append(
                        episode_next_states_np.reshape(-1, *episode_next_states_np.shape[2:])
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
        gc.collect()
        
        return all_states, all_returns, all_next_states, all_episodic_returns

    def train_baseline_state_aggregation_module(self):
        self.autoencoder.train()
        optimizer = optim.Adam(
            self.autoencoder.parameters(), lr=self.autoencoder_learning_rate
        )
        
        with torch.no_grad():
            states_np, returns_np, next_states_np, episodic_returns = (
                self.generate_rollouts(self.state_aggregation_baseline_num_rollout)
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
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.autoencoder_batch_size,
            shuffle=False,
            generator=torch.Generator(device=device),
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.autoencoder_batch_size,
            shuffle=False,
            generator=torch.Generator(device=device),
            pin_memory=True,
        )

        for epoch in range(self.autoencoder_epochs):
            if self.print_logs:
                print(
                    f"Training State Aggregation Module - Epoch: {epoch + 1}/{self.autoencoder_epochs}"
                )

            for batch in train_loader:
                current_state, _, next_states, episodic_return = batch
                current_state = current_state.to(device)
                next_states = next_states.to(device)
                episodic_return = episodic_return.to(device)
                current_state.requires_grad = True

                encoder_out, current_state_pred, next_state_pred = self.autoencoder(
                    current_state
                )
                loss = self.autoencoder.compute_loss(
                    next_state_pred,
                    next_states,
                    current_state_pred,
                    current_state,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            writer.add_scalar(
                "Loss/state_aggregation_autoencoder_loss",
                loss.item(),
                self.state_aggregation_autoencoder_loss_counter,
            )
            self.state_aggregation_autoencoder_loss_counter += 1

            if epoch % self.autoencoder_validation_frequency == 0:
                self.autoencoder.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        current_state, _, next_states, episodic_return = batch
                        current_state = current_state.to(device)
                        next_states = next_states.to(device)
                        episodic_return = episodic_return.to(device)
                        encoder_out, current_state_pred, next_state_pred = self.autoencoder(
                            current_state
                        )
                        loss = self.autoencoder.compute_loss(
                            next_state_pred,
                            next_states,
                            current_state_pred,
                            current_state,
                        )
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                writer.add_scalar("Loss/validation_autoencoder_loss", val_loss, epoch)
                self.autoencoder.train()

        self.autoencoder.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                current_state, _, next_states, episodic_return = batch
                current_state = current_state.to(device)
                next_states = next_states.to(device)
                episodic_return = episodic_return.to(device)
                encoder_out, current_state_pred, next_state_pred = self.autoencoder(
                    current_state
                )
                loss = self.autoencoder.compute_loss(
                    next_state_pred,
                    next_states,
                    current_state_pred,
                    current_state,
                )
                test_loss += loss.item()
        test_loss /= len(test_loader)

        if self.print_logs:
            print(f"Final Test Loss: {test_loss}")
        
        self.autoencoder.save(run_name=run_name)
        self.autoencoder.eval()

    def train_single_step_state_aggregation_module(
        self, observations, next_observations
    ):
        with torch.no_grad():
            next_states = torch.tensor(
                next_observations, dtype=torch.float32, device=device
            )
            current_state = torch.tensor(
                observations, dtype=torch.float32, device=device
            )
        
        self.autoencoder.train()
        optimizer = torch.optim.Adam(
            lr=self.autoencoder_learning_rate, params=self.autoencoder.parameters()
        )

        _, current_state_pred, next_state_pred = self.autoencoder(
            current_state
        )
        loss = self.autoencoder.compute_loss(
            next_state_pred,
            next_states,
            current_state_pred,
            current_state,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if self.global_step % 100 == 0:
            writer.add_scalar(
                "Loss/state_aggregation_autoencoder_loss",
                loss.item(),
                self.global_step,
            )

        self.autoencoder.eval()
        
    def train_single_step_exploration_critic(self, latent_state, action, latent_next_state, next_state):
        
        if len(self.exploration_buffer) < self.exploration_buffer_num_experiences:
            if self.print_logs:
                print(f"Training Exploration Critic: Insufficient experience: Current - {len(self.exploration_buffer)}, Need - {self.exploration_buffer_num_experiences}")
            return

        self.exploration_critic.train()
        optimizer = optim.Adam(
            list(self.exploration_critic.parameters()), lr=self.exploration_critic_learning_rate
        )

        with torch.no_grad():
            samples = self.exploration_buffer.sample_from_distance(
                self.exploration_critic_batch_size * self.exploration_buffer_num_experiences, latent_state
            )
            
            # Handle empty buffer case
            if samples is None:
                return

            # Use the pre-stacked tensors directly from samples
            buffer_latent_states = samples.latent_states
            buffer_actions = samples.actions
            buffer_latent_next_states = samples.latent_next_states
            
            if latent_state.dim() == 1:
                latent_state = latent_state.unsqueeze(0)
            if action.dim() == 1:
                action = action.unsqueeze(0)                
            if latent_next_state.dim() == 1:
                latent_next_state = latent_next_state.unsqueeze(0)
            if next_state.dim() == 1:
                next_state = next_state.unsqueeze(0)
            
            combined_next_states = next_state.repeat(self.exploration_critic_batch_size, 1)
            combined_latent_states = latent_state.repeat(self.exploration_critic_batch_size, 1)
            combined_actions = action.repeat(self.exploration_critic_batch_size, 1)
            combined_latent_next_states = latent_next_state.repeat(self.exploration_critic_batch_size, 1)
                        
            buffer_seq = torch.cat([
                buffer_latent_states,
                buffer_actions,
                buffer_latent_next_states
            ], dim=-1).reshape(self.exploration_critic_batch_size, self.exploration_buffer_num_experiences, -1)
        
        _, decoder_next_states = self.autoencoder.decoder(
            combined_latent_next_states
        )

        loss = self.exploration_critic.compute_loss(
            latent_state=combined_latent_states,
            action=combined_actions,
            latent_next_state=combined_latent_next_states,
            exploration_buffer_seq=buffer_seq,
            next_state=combined_next_states,
            decoder_next_state=decoder_next_states
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/exploration_critic_loss", loss.item(), self.global_step)
        
        self.exploration_critic.eval()
        
    def train(self):
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
        
        max_episode_length = envs.envs[0].spec.max_episode_steps
        current_episode_states = np.zeros((max_episode_length, *obs.shape[1:]), dtype=np.float32)
        current_episode_rewards = np.zeros((max_episode_length, 1), dtype=np.float32)
        current_episode_next_states = np.zeros((max_episode_length, *obs.shape[1:]), dtype=np.float32)
        episode_step = 0
        
        noise_tensor = torch.zeros_like(self.actor(
            self.autoencoder.encoder(torch.tensor(obs, dtype=torch.float32, device=device))
        ))
        
        for global_step in range(self.actor_critic_timesteps):
            self.global_step = global_step
            encoded_obs = self.autoencoder.encoder(torch.tensor(obs, dtype=torch.float32, device=device)).detach().clone()
            if global_step < self.learning_starts:
                actions = np.array(
                    [envs.single_action_space.sample() for _ in range(envs.num_envs)]
                )
            else:
                actions = self.actor(encoded_obs).detach()
                # noise_tensor = torch.randn_like(actions) * self.actor.action_scale * self.exploration_noise
                actions = (actions + noise_tensor).cpu().numpy().clip(
                    self.env.single_action_space.low, self.env.single_action_space.high
                )

            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            encoded_next_obs = self.autoencoder.encoder(torch.tensor(next_obs, dtype=torch.float32, device=device)).detach().clone()
                
            self.exploration_buffer.add(
                encoded_obs.squeeze(0), 
                torch.tensor(actions, dtype=torch.float32, device=device).squeeze(0), 
                encoded_next_obs.squeeze(0),
                torch.tensor(next_obs, dtype=torch.float32, device=device).squeeze(0)
            )

            if global_step % self.exploration_critic_update_frequency == 0:
                self.train_single_step_exploration_critic(
                    encoded_obs,
                    torch.tensor(actions, dtype=torch.float32, device=device).squeeze(0),
                    encoded_next_obs,
                    torch.tensor(next_obs, dtype=torch.float32, device=device).squeeze(0)
                )
            
            if episode_step < max_episode_length:
                current_episode_states[episode_step] = obs.squeeze(0)
                current_episode_rewards[episode_step] = rewards.squeeze(0)
                current_episode_next_states[episode_step] = next_obs.squeeze(0)
                episode_step += 1
            
            if "final_info" in infos:
                for reward in infos["final_info"]["episode"]["r"]:
                    if self.print_logs:
                        print(f"global_step={global_step}, episodic_return={reward}")
                    writer.add_scalar("charts/episodic_return", reward, global_step)
                    break

                for length in infos["final_info"]["episode"]["l"]:
                    writer.add_scalar("charts/episodic_length", length, global_step)
                    break

            real_next_obs = next_obs.copy()
            for idx, is_done in enumerate(np.logical_or(terminations, truncations)):
                if is_done:
                    real_next_obs[idx] = infos["final_obs"][idx]
                    batch = self.replay_buffer.sample(self.autoencoder_batch_size)
                    self.train_single_step_state_aggregation_module(
                        torch.tensor(batch.observations), 
                        torch.tensor(batch.next_observations)
                    )
                    episode_step = 0

            self.replay_buffer.add(
                obs, real_next_obs, actions, rewards, terminations, infos,
                encoded_obs=encoded_obs, 
                encoded_next_obs=encoded_next_obs
            )

            obs = next_obs
            
            # Actor-critic training
            if self.global_step > self.learning_starts:
                data = self.replay_buffer.sample(self.actor_critic_batch_size)
                
                with torch.no_grad():
                    next_state_actions = self.actor_target(self.autoencoder.encoder(data.next_observations))
                    qf1_next_target = self.critic_target(data.next_observations, next_state_actions)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * qf1_next_target.view(-1)
                
                qf1_a_values = self.critic(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                
                critic_optimizer.zero_grad()
                qf1_loss.backward()
                critic_optimizer.step()

                if global_step % self.policy_update_frequency == 0:
                    latent_observations = self.autoencoder.encoder(data.observations)
                    latent_next_observations = self.autoencoder.encoder(data.next_observations)
                    actor_action = self.actor(latent_observations)
                    critic_value = self.critic(data.observations, actor_action).mean()

                    exploration_buffer_seq = torch.cat([
                        data.trajectory.latent_states,
                        data.trajectory.actions,
                        data.trajectory.latent_next_states
                    ], dim=-1)

                    exploration_score = self.exploration_critic(
                        latent_observations,
                        actor_action,
                        latent_next_observations,
                        exploration_buffer_seq
                    ).mean()
                    
                    if self.global_step % 100 == 0:
                        writer.add_scalar("charts/exploration_score", exploration_score.item(), self.global_step)
                    
                    actor_loss = -critic_value * (1 + exploration_score)
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    
                    # Restore original logging
                    if self.print_logs:
                        print(f"Training Actor Critic: Current timestep: {global_step}/{self.actor_critic_timesteps}, Actor Loss: {actor_loss.item()}, Critic Loss: {qf1_loss.item()}")

                    # update the target network
                    for param, target_param in zip(
                        self.actor.parameters(), self.actor_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.polyak_constant * param.data + (1 - self.polyak_constant) * target_param.data
                        )
                    for param, target_param in zip(
                        self.critic.parameters(), self.critic_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.polyak_constant * param.data + (1 - self.polyak_constant) * target_param.data
                        )
                if self.global_step % 100 == 0:
                    writer.add_scalar("Loss/actor_loss", actor_loss.item(), self.global_step)
                    writer.add_scalar("Loss/qf1_values", qf1_a_values.mean().item(), self.global_step)
                    writer.add_scalar("Loss/critic_loss", qf1_loss.item(), self.global_step)
                
            
        self.env.close()
        self.actor.save(run_name=run_name)
        self.critic.save(run_name=run_name)
        self.exploration_critic.save(run_name=run_name)
        self.autoencoder.save(run_name=run_name)


if __name__ == "__main__":
    ecn_trainer = ECNTrainer(
        envs,
        print_logs=False
    )

    ecn_trainer.train()
    evaluate_ecn(
        autoencoder_path=f"{run_name}/models/state_aggregation_autoencoder.pt",
        actor_path=f"{run_name}/models/actor.pt",
        make_env=make_env,
        env_id=args.env_id,
        eval_episodes=10,
        run_name=run_name,
        device=device,
        exploration_noise=args.exploration_noise
    )
