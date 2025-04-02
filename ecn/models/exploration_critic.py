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
        if not has_batch:
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
            return torch.abs(exploration_score).squeeze(0)
        return torch.abs(exploration_score)
    
        
    def compute_loss(self, latent_state, action, latent_next_state, exploration_buffer_seq):
        """
        Compute a loss for the exploration critic. Creates a conditional next state distribution based on the exploration buffer sequence.
        Computes the probability of the next state given the current state and action under the conditional distribution.
        Subtracts this probability from 1 to get the exploration score (higher probability -> less novelty).

        latent_state: Latent state representation, shape (batch_size, latent_dim)
        action: Action taken, shape (batch_size, action_dim)
        latent_next_state: latent next state, shape (batch_size, latent_dim)
        exploration_buffer_seq: Sequence of past experiences, shape (batch_size, seq_len, latent_dim + action_dim + latent_dim)

        Returns: mean loss value across the batch
        """
        # Extract components from exploration buffer
        latent_dim = latent_state.shape[1]
        action_dim = action.shape[1]
        batch_size = latent_state.shape[0]
        seq_len = exploration_buffer_seq.shape[1]
        
        # Split exploration buffer into state, action, next_state components
        buffer_states = exploration_buffer_seq[:, :, :latent_dim]  # [B, seq_len, latent_dim]
        buffer_actions = exploration_buffer_seq[:, :, latent_dim:latent_dim+action_dim]  # [B, seq_len, action_dim]
        buffer_next_states = exploration_buffer_seq[:, :, latent_dim+action_dim:]  # [B, seq_len, latent_dim]
        
        # Reshape buffer tensors to apply embeddings in a single operation
        # Flatten batch and sequence dimensions
        flat_buffer_states = buffer_states.reshape(batch_size * seq_len, -1)  # [B*seq_len, latent_dim]
        flat_buffer_actions = buffer_actions.reshape(batch_size * seq_len, -1)  # [B*seq_len, action_dim]
        
        # Apply embeddings
        embedded_state = self.state_embedding(latent_state)  # [B, embed_dim]
        embedded_action = self.action_embedding(action)  # [B, embed_dim]
        
        flat_embedded_buffer_states = self.state_embedding(flat_buffer_states)  # [B*seq_len, embed_dim]
        flat_embedded_buffer_actions = self.action_embedding(flat_buffer_actions)  # [B*seq_len, embed_dim]
        
        # Reshape back to [B, seq_len, embed_dim]
        embedded_buffer_states = flat_embedded_buffer_states.reshape(batch_size, seq_len, -1)
        embedded_buffer_actions = flat_embedded_buffer_actions.reshape(batch_size, seq_len, -1)
        
        # Compute similarity kernel between current state-action and buffer state-actions
        # First concatenate the embeddings
        current_sa_embedding = torch.cat([embedded_state, embedded_action], dim=1)  # [B, 2*embed_dim]
        buffer_sa_embeddings = torch.cat([embedded_buffer_states, embedded_buffer_actions], dim=2)  # [B, seq_len, 2*embed_dim]
        
        # Reshape for broadcasting
        current_sa_embedding = current_sa_embedding.unsqueeze(1)  # [B, 1, 2*embed_dim]
        
        # Compute L2 distance (squared Euclidean distance)
        distances = torch.sum((current_sa_embedding - buffer_sa_embeddings)**2, dim=2)  # [B, seq_len]
        
        # Apply RBF kernel: K(x,y) = exp(-||x-y||^2 / (2*bandwidth^2))
        bandwidth = 1.0  # Hyperparameter that can be tuned
        kernel_weights = torch.exp(-distances / (2 * bandwidth**2))  # [B, seq_len]
        
        # Normalize weights to sum to 1
        kernel_weights = kernel_weights / (torch.sum(kernel_weights, dim=1, keepdim=True) + 1e-10)  # [B, seq_len]
        
        # Now compute probability of next state using KDE
        # Compute distances between current next_state and buffer next_states
        current_next_state_emb = latent_next_state.unsqueeze(1)  # [B, 1, latent_dim]
        next_state_distances = torch.sum((current_next_state_emb - buffer_next_states)**2, dim=2)  # [B, seq_len]
        
        # Apply RBF kernel for next state similarity
        next_state_bandwidth = 1.0  # Can be different from state-action bandwidth
        next_state_kernel = torch.exp(-next_state_distances / (2 * next_state_bandwidth**2))  # [B, seq_len]
        
        # This represents P(next_state | state, action) estimated by KDE
        conditional_density = torch.sum(kernel_weights * next_state_kernel, dim=1)  # [B]
        conditional_prob = torch.sigmoid(conditional_density)  # [B]
        
        target_exploration_score = 1.0 - conditional_prob  # [B]
        exploration_score = self.forward(latent_state, action, latent_next_state, exploration_buffer_seq)  # [B, 1]
        exploration_score = exploration_score.squeeze(1)  # [B]
        
        loss = torch.abs(exploration_score - target_exploration_score).sum()
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