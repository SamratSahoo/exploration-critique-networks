from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ecn.models.state_aggregation_autoencoder import StateAggregationAutoencoder
from ecn.models.actor import Actor

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    exploration_noise: float = 0.1,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, 0, capture_video, run_name, True)],
        autoreset_mode=gym.vector.vector_env.AutoresetMode.SAME_STEP
    )
    actor = Model[0](envs).to(device)
    qf = Model[1](envs).to(device)
    actor_params, qf_params = torch.load(model_path, map_location=device)
    actor.load_state_dict(actor_params)
    actor.eval()
    qf.load_state_dict(qf_params)
    qf.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            actions += torch.normal(0, actor.action_scale * exploration_noise)
            actions = (
                actions.cpu()
                .numpy()
                .clip(envs.single_action_space.low, envs.single_action_space.high)
            )

        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for reward in infos["final_info"]["episode"]["r"]:
                print(f"eval_episode={len(episodic_returns)}, episodic_return={reward}")
                episodic_returns.append(reward)
                break
        obs = next_obs

    return episodic_returns


def evaluate_ecn(
    autoencoder_path: str,
    actor_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    exploration_noise: float = 0.1,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, 0, capture_video, run_name, True)],
        autoreset_mode=gym.vector.vector_env.AutoresetMode.SAME_STEP
    )
    
    latent_size = np.array(envs.single_observation_space.shape).prod() // 2
    
    # Load state dictionaries
    actor = torch.load(actor_path, map_location=device, weights_only=False)
    actor.eval()

    autoencoder_state_dict = torch.load(autoencoder_path, map_location=device, weights_only=False)
    autoencoder = StateAggregationAutoencoder(envs, latent_size=latent_size)
    autoencoder.load_state_dict(autoencoder_state_dict)
    autoencoder.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            latent_obs = autoencoder.encoder(torch.Tensor(obs).to(device))
            actions = actor(latent_obs)
            actions += torch.normal(0, actor.action_scale * exploration_noise)
            actions = (
                actions.cpu()
                .numpy()
                .clip(envs.single_action_space.low, envs.single_action_space.high)
            )

        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for reward in infos["final_info"]["episode"]["r"]:
                print(f"eval_episode={len(episodic_returns)}, episodic_return={reward}")
                episodic_returns.append(reward)
                break
        obs = next_obs
        print(episodic_returns)
    return episodic_returns