import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, max_positions: int = 512, init_cache: bool = True):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.max_positions = max_positions
        if init_cache:
            cache = self._init_cos_sin_cache()
            self.register_buffer("cos_sin_cache", cache, persistent=False)


    def _compute_inv_freqs(self, rotary_dim: int, base: int = 10000):
        # Compute the inverse frequencies for the rotary embedding
        # The frequencies are determined by the formula: 1 / (10000^(2i/rotary_dim)) for i in [0, rotary_dim//2)
        return 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

    def _init_cos_sin_cache(self):
        # Initialize the cosine and sine cache for the rotary embedding
        inv_freqs = self._compute_inv_freqs(self.rotary_dim)
        t = torch.arange(self.max_positions, dtype=torch.float)
        # Compute the frequencies for each position and each dimension using the outer product of positions and inverse frequencies
        # [max_positions, rotary_dim // 2], each position has a unique set of frequencies for each dimension
        freqs = torch.einsum("i,j->ij", t, inv_freqs)
        cos = freqs.cos()
        sin = freqs.sin()
        return torch.cat((cos, sin), dim=-1)

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, head_size: int, is_neox_style: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
        # Apply the rotary embedding to the query and key tensors based on the provided positions
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        # [max_positions, rotary_dim // 2] -> [num_tokens, 1, rotary_dim // 2]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        # [max_positions, rotary_dim], where the first half contains cosine values and the second half contains sine values for each position
        cos, sin = torch.chunk(cos_sin, 2, dim=-1)
        # Reshape cos and sin to match the dimensions of query and key tensors for element-wise multiplication
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        q_shape = query.shape
        k_shape = key.shape
        
        query = query.view(num_tokens, -1, head_size)
        q_rot = query[..., :self.rotary_dim]
        q_pass = query[..., self.rotary_dim:]
        
        key = key.view(num_tokens, -1, head_size)
        k_rot = key[..., :self.rotary_dim]
        k_pass = key[..., self.rotary_dim:]

        if is_neox_style:
            q_rot = self._apply_rotary_embedding_neox_style(cos, sin, q_rot)
            k_rot = self._apply_rotary_embedding_neox_style(cos, sin, k_rot)
        else:
            q_rot = self._apply_rotary_embedding(cos, sin, q_rot)
            k_rot = self._apply_rotary_embedding(cos, sin, k_rot)

        query_rotated = torch.cat((q_rot, q_pass), dim=-1).reshape(q_shape)
        key_rotated = torch.cat((k_rot, k_pass), dim=-1).reshape(k_shape)
        return query_rotated, key_rotated


    def _apply_rotary_embedding_neox_style(self, cos: torch.Tensor, sin: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Apply the rotary embedding transformation in the style used by NeoX
        # This involves interleaving the cosine and sine transformations across the dimensions of x
        # [cos(p·fi)  -sin(p·fi)] [x1_i]
        # [sin(p·fi)   cos(p·fi)] [x2_i]
        # The input tensor is splitted into two parts along the last dimension, where x1 contains the even indexed dimensions and x2 contains the odd indexed dimensions
        x1, x2 = torch.chunk(x, 2, dim=-1)
        # x1/x2 shape: [num_tokens, num_heads, rotary_dim // 2], cos/sin shape: [num_tokens, 1, rotary_dim // 2], do elements-wise matmul.
        o1 = cos * x1 - sin * x2
        o2 = sin * x1 + cos * x2
        return torch.cat((o1, o2), dim=-1)


    def _apply_rotary_embedding(self, cos: torch.Tensor, sin: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Apply the rotary embedding transformation to the input tensor x using the cosine and sine values
        # The transformation is defined as:
        # [cos(p·fi)  -sin(p·fi)] [x1_i]
        # [sin(p·fi)   cos(p·fi)] [x2_i]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        o1 = cos * x1 - sin * x2
        o2 = sin * x1 + cos * x2
        return torch.stack((o1, o2), dim=-1).flatten(-2)
