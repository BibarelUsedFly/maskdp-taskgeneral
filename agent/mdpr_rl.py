import torch
import torch.nn as nn
from agent.mdpr import MaskedDP

class ActorR(nn.Module):
    def __init__(self, obs_dim, action_dim, attention_length, config, reward_dim=1):
        super().__init__()
        self.n_embd = config.n_embd
        self.max_len = attention_length
        self.model = MaskedDP(obs_dim, action_dim, config, reward_dim)
        self.ln = nn.LayerNorm(self.n_embd)
        self.register_buffer("attn_mask",
            torch.tril(torch.ones(self.max_len, self.max_len)) \
            [None, None, ...],) # (1, 1, 3*T, 3*T)
        # The differece with the original model mask is that here we use
        # a triangular matrix so that attention is causal and doesn't cheat

    def forward(self, obs_seq):
        # (batch_size = 1,
        #  max_length (T) = agent.transformer_cfg.traj_length = 64,
        #  obs_dim = 24)
        batch_size, T, obs_dim = obs_seq.size() 

        # This is the linear state embedding layer
        state = self.model.state_embed(obs_seq) # (B, T, 24) -> (B, T, 256)
        state += self.model.timestep_embed[:, 0::3] # Only for S tokens here 
        state += self.model.modality_embed.weight[0]

        B, T, D = state.shape
        # Expand mask_tokens from 1, 1, D -> B, T, D
        mask_tok = self.model.mask_token.expand(B, T, D)

        a = mask_tok.clone()
        a = a + self.model.timestep_embed[:, 1::3]
        a = a + self.model.modality_embed.weight[1]

        r = mask_tok.clone()
        r = r + self.model.timestep_embed[:, 2::3]
        r = r + self.model.modality_embed.weight[2]

        # Interleave into [s0, a0, r0, s1, a1, r1, ...] -> (B, 3T, D)
        x = torch.stack([state, a, r], dim=2).reshape(B, 3*T, D)

        for blk in self.model.encoder_blocks:
            x = blk(x, self.attn_mask)
        x = self.model.encoder_norm(x)

        s = self.model.decoder_state_embed(x[:, 0::3]) + self.model.modality_embed.weight[0]
        a = self.model.decoder_action_embed(x[:, 1::3]) + self.model.modality_embed.weight[1]
        r = self.model.decoder_reward_embed(x[:, 2::3]) + self.model.modality_embed.weight[2]

        x = torch.stack([s, a, r], dim=1).permute(0, 2, 1, 3).reshape_as(x)
        x = x + self.model.decoder_timestep_embed
        
        for blk in self.model.decoder_blocks:
            x = blk(x, self.attn_mask)

        # predictor projection
        a = self.model.action_head(x[:, 1::3])
        predicted_action = a[:, -1] # (1, action_dim) = (1, 6)
        return predicted_action


# self.model.pos_embed.shape = (1, 192, 256)
#      = (batch_size, 3*traj_length, n_embd)

# self.model.timestep_embed.shape = (1, 192, 256)
#      = (batch_size, 3*traj_length, n_embd)
# This is a 3x repeat of a traj_length timestep embedding

# self.model.modality_embed.weight[0].shape = (256,)
#      = (n_embd)
