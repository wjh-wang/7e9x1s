import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UdNet(nn.Module):
    def __init__(self, config):
        super().__init__()
    
        hidden_size=128
        self.hidden_size = hidden_size
        self.temb_ch = hidden_size * 4
        

        self.temb = nn.ModuleList([
        nn.Linear(hidden_size, hidden_size * 2),
        nn.Linear(hidden_size * 2, hidden_size * 2),
        ]   )
        # Downsampling
        self.down = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size * 2)
        ])

        # Middle block with Cross-Attention
        self.mid_block = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=4, batch_first=True)

        # Upsampling
        self.up = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size * 3)  # predict 3 outputs

    def get_timestep_embedding(self, timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, v_noisy, p_noisy, t_noisy, t):
        # timestep embedding
        temb = self.get_timestep_embedding(t, self.hidden_size)
        temb = F.silu(self.temb[0](temb))
        temb = self.temb[1](temb)

        # Downsampling（MLP）
        h = F.silu(self.down[0](v_noisy))
        h = F.silu(self.down[1](h + temb))

        # Middle block with Cross-Attention
        condition = torch.cat([p_noisy, t_noisy], dim=1).unsqueeze(1)  # [B, 1, 2*hidden_size]
        h_flat = h.unsqueeze(1)  # [B, 1, hidden_dim]
        h_flat, _ = self.cross_attn(h_flat, condition, condition)
        h = h + h_flat.squeeze(1)
        h = F.silu(self.mid_block(h))

        # Upsampling（MLP）
        h = F.silu(self.up[0](h + temb))
        h = F.silu(self.up[1](h))

        # Output projection
        out = self.output_proj(h)
        predicted_v1, predicted_v2, predicted_v3 = out.chunk(3, dim=1)
        return predicted_v1, predicted_v2, predicted_v3
