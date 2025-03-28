import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

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

    def save(self, run_name=None, path=None):
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

    def save(self, run_name=None, path=None):
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
        # Set batch normalization to eval mode during forward passes if not training
        train_mode = self.training
        if not train_mode:
            self.encoder.eval()
            self.decoder.eval()
        
        encoder_out = self.encoder(state)
        current_state_pred, next_state_pred = self.decoder(encoder_out)
        
        # Restore original training mode
        if not train_mode:
            self.encoder.train(train_mode)
            self.decoder.train(train_mode)
        
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
        # Clone tensors to avoid in-place modifications
        current_state_clone = current_state.clone()
        next_states_clone = next_states.clone()
        
        loss_current_state = F.mse_loss(current_state_pred, current_state_clone)
        loss_next_state = F.mse_loss(next_state_pred, next_states_clone)
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

    def save(self, run_name=None, path=None):

        if path:
            self.encoder.save(path=path)
            self.decoder.save(path=path)
            torch.save(self.state_dict(), path)
        else:
            self.encoder.save(run_name=run_name)
            self.decoder.save(run_name=run_name)
            torch.save(
                self.state_dict(), f"{run_name}/models/state_aggregation_autoencoder.pt"
            )

