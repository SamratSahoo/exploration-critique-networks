import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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
        # Create a new tensor instead of modifying in-place
        result = exploration_buffer_seq + self.positional_encoding[
            : exploration_buffer_seq.size(0)
        ]
        return self.dropout(result)

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
        self.max_seq_len = max_seq_len
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
        # Check if inputs have batch dimension
        has_batch = latent_state.dim() > 1
        
        # Handle inputs with shape flexibility
        if not has_batch:
            # Add batch dimension if not present
            latent_state = latent_state.unsqueeze(0)
            action = action.unsqueeze(0)
            latent_next_state = latent_next_state.unsqueeze(0)
            if exploration_buffer_seq.dim() == 2:
                exploration_buffer_seq = exploration_buffer_seq.unsqueeze(0)
        
        current_state_token = self.state_embedding(latent_state)         # [B, embed_dim]
        action_token = self.action_embedding(action)                     # [B, embed_dim]
        next_state_token = self.state_embedding(latent_next_state)       # [B, embed_dim]
        
        expl_buffer_token = self.exploration_embedding(exploration_buffer_seq)  # [B, seq_len, embed_dim]
        expl_buffer_token = expl_buffer_token.transpose(0, 1) # [seq_len, B, embed_dim]

        pos_expl_buffer_token = self.positional_encoding(expl_buffer_token)
        # query tensor: [3, B, embed_dim]
        query = torch.stack([current_state_token, action_token, next_state_token], dim=0)
        cross_attention_out = self.cross_attention(query, pos_expl_buffer_token)  # [3, B, embed_dim]
        aggregate_cross = cross_attention_out.mean(dim=0)  # [B, embed_dim]
        exploration_score = self.fc_out(aggregate_cross)  # [B, 1]
        
        if not has_batch:
            return exploration_score.squeeze(0)
        return exploration_score

    def compute_loss(self, latent_state, action, latent_next_state, exploration_buffer_seq):
        exploration_score = self(latent_state, action, latent_next_state, exploration_buffer_seq)
        
        # Make sure latent_state has batch dimension
        if latent_state.dim() == 1:
            latent_state = latent_state.unsqueeze(0)
            action = action.unsqueeze(0)
            latent_next_state = latent_next_state.unsqueeze(0)
        
        # Check if exploration buffer has elements and correct dimensions
        if exploration_buffer_seq.dim() == 3 and exploration_buffer_seq.shape[1] > 0:
            # Extract components from exploration buffer
            latent_dim = latent_state.shape[-1]
            action_dim = action.shape[-1]
            
            # Extract states, actions, and next states from buffer
            buffer_states = exploration_buffer_seq[:, :, :latent_dim]
            buffer_actions = exploration_buffer_seq[:, :, latent_dim:latent_dim+action_dim]
            buffer_next_states = exploration_buffer_seq[:, :, -latent_dim:]
            
            # Prepare current transition and action for comparison
            current_transition = latent_next_state - latent_state  # [B, latent_dim]
            current_transition = current_transition.unsqueeze(1)  # [B, 1, latent_dim]
            current_action = action.unsqueeze(1)  # [B, 1, action_dim]
            
            # Calculate state transition differences (vectorized)
            buffer_transitions = buffer_next_states - buffer_states  # [B, seq_len, latent_dim]
            transition_diffs = torch.linalg.norm(current_transition - buffer_transitions, dim=-1)  # [B, seq_len]
            
            # Calculate action differences (vectorized)
            action_diffs = torch.linalg.norm(current_action - buffer_actions, dim=-1)  # [B, seq_len]
            
            # Normalize differences to balance their influence
            norm_transition_diffs = transition_diffs / transition_diffs.clamp(min=1e-6).mean()
            norm_action_diffs = action_diffs / action_diffs.clamp(min=1e-6).mean()
            
            # Combined similarity using both state transitions and actions
            # Higher combined_diffs means lower similarity
            combined_diffs = norm_transition_diffs + norm_action_diffs
            similarities = 1.0 / (1.0 + combined_diffs)  # [B, seq_len]
            
            # Only use valid buffer entries (up to max_seq_len)
            valid_length = min(self.max_seq_len, exploration_buffer_seq.shape[1])
            similarities = similarities[:, :valid_length]
            
            # Average similarity across buffer entries
            similarity = similarities.mean(dim=1, keepdim=True)  # [B, 1]
            
            # Target: high novelty = low similarity = high score
            novelty_target = 1.0 - similarity
            loss = F.mse_loss(exploration_score, novelty_target)
        else:
            loss = -exploration_score.mean()
        
        return loss
    
    def run_inference(self, encoded_data_obs, actor_action, encoded_data_next_obs, trajectory):
        """
        Run inference using trajectory data from the TrajectoryReplayBuffer
        
        Args:
            encoded_data_obs: Encoded observations (batch_size, obs_dim)
            actor_action: Actions produced by the actor (batch_size, action_dim)
            encoded_data_next_obs: Encoded next observations (batch_size, obs_dim)
            trajectory: Trajectory object containing latent_states, actions, latent_next_states
                        Each with shape (batch_size, seq_len, dim)
                        
        Returns:
            Mean exploration critic score for the batch
        """
        if encoded_data_obs.shape[0] == 0:
            return torch.tensor(0.0, device=encoded_data_obs.device)
        
        # Create the exploration buffer sequence for the entire batch
        # Shape: [batch_size, seq_len, latent_dim + action_dim + latent_dim]
        exploration_buffer_seq = torch.cat([
            trajectory.latent_states,       # [batch_size, seq_len, latent_dim]
            trajectory.actions,             # [batch_size, seq_len, action_dim]
            trajectory.latent_next_states   # [batch_size, seq_len, latent_dim]
        ], dim=2)
        
        # Process the entire batch at once
        scores = self(
            encoded_data_obs,
            actor_action,
            encoded_data_next_obs,
            exploration_buffer_seq
        )
        
        # Return mean score across the batch
        return torch.mean(scores)

    def save(self, run_name=None, path=None):
        if not path:
            torch.save(self.state_dict(), f"{run_name}/models/exploration_critic.pt")
        else:
            torch.save(self.state_dict(), path)