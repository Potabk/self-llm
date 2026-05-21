# norm算子
import torch
from torch import nn


# RMSNorm:   output = x / sqrt(E[x²] + ε)  * weight
# LayerNorm: output = (x - E[x]) / sqrt(Var[x] + ε) * gamma + beta


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self, x: torch.Tensor):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        output = x * torch.rsqrt(variance + self.eps)
        return output * self.gamma

class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
    

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        output = (x - mean) * torch.rsqrt(variance + self.eps)

        return output * self.gamma + self.beta
        
        
        
        
if __name__ == '__main__':
    input = torch.randn(2, 10, 512)  # [batch_size, seq_len, hidden_size]
    norm = LayerNorm(512)
    output = norm(input)
    print(output.shape)  # [batch_size, seq_len, variance_size]
    #print(output)
