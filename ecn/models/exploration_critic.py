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
        current_state_token = self.state_embedding(latent_state)
        action_token = self.action_embedding(action)
        next_state_token = self.state_embedding(latent_next_state)
        expl_buffer_token = self.exploration_embedding(exploration_buffer_seq)

        if expl_buffer_token.dim() == 2:
            # Add batch dimension
            expl_buffer_token = expl_buffer_token.unsqueeze(1)
        else:
            # Swap seq len and batch dimension
            expl_buffer_token = expl_buffer_token.transpose(0, 1)

        pos_expl_buffer_token = self.positional_encoding(expl_buffer_token.clone())
        query = torch.stack([current_state_token, action_token, next_state_token], dim=0)
        
        if query.dim() == 2:
            cross_attention_out = self.cross_attention(query, pos_expl_buffer_token.squeeze(1))
        else:
            query = query.transpose(0, 1)
            pos_expl_buffer_token = pos_expl_buffer_token.transpose(0, 1)
            cross_attention_out = self.cross_attention(query, pos_expl_buffer_token)
        aggregate_cross = cross_attention_out.mean(dim=0)
        
        exploration_score = self.fc_out(aggregate_cross)
        return exploration_score

    def run_inference(self, encoded_data_obs, actor_action, encoded_data_next_obs, recent_experiences):
        if len(recent_experiences) >= self.max_seq_len:
            buffer_seq_list = []
            for exp in recent_experiences:
                concatenated = torch.cat([exp.latent_state, exp.action, exp.latent_next_state], dim=-1)
                buffer_seq_list.append(concatenated)
            exploration_buffer_seq = torch.stack(buffer_seq_list)
            exploration_critic_score = self(
                encoded_data_obs, 
                actor_action, 
                encoded_data_next_obs, 
                exploration_buffer_seq.unsqueeze(0)
            ) 
        else:
            exploration_critic_score = torch.scalar_tensor(0)
        
        return exploration_critic_score


    def save(self, run_name=None, path=None):
        if not path:
            torch.save(self.state_dict(), f"{run_name}/models/exploration_critic.pt")
        else:
            torch.save(self.state_dict(), path)