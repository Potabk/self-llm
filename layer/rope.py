import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, init_cache: bool = True, is_neox_style: bool=False, max_position_embedding: int = 512):
        super().__init__()
        self.dim = dim
        self.max_position_embedding = max_position_embedding
        self.is_neox_style = is_neox_style
        if init_cache:
            cache = self._init_cos_sin_cache(dim)  # Default max sequence length
            self.register_buffer("cos_cache", cache, persistent=False)
            

    def _compute_inv_frequency(self, rotary_dim: int, base: int = 10000):
        inv_freq = 1.0 / (torch.arange(0, rotary_dim, 2).float() / rotary_dim * base)
        return inv_freq
    

    def _init_cos_sin_cache(self, rotary_dim: int):
        inv_freq = self._compute_inv_frequency(rotary_dim)
        t = torch.arange(self.max_position_embedding, dtype=torch.float)
        # [max_position_embedding, rotary_dim // 2], each position has a unique set of frequencies for each req
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos_cache = freqs.cos()
        sin_cache = freqs.sin()
        return torch.cat(cos_cache, sin_cache, dim=-1)
    
    def get_cos_sin(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_cache[:seq_len]
        cos, sin = torch.chunk(cos_sin, 2, dim=-1)
        return cos, sin

    def forward(self, positions: torch.Tensor, 
                query: torch.Tensor,
                key: torch.Tensor,
                head_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        
