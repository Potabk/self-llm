# 实现多个激活函数

import torch
from torch import nn



_ACTIVATION_REGISTRY = {
        "gelu": lambda: Gelu(),
        "silu": lambda: nn.SiLU(),
    }


class Gelu(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward():
        pass

class SiLu(nn.Module):
    def __init__(self):
        super().__init__()

    # silu(x) = x * sigmoid(x)
    #      = x * (1 / (1 + e^(-x)))
    def forward_native(self, x: torch.Tensor):
        return x / (1 + torch.exp(-x))

    def forward_torch(self, x: torch.Tensor):
        return torch.nn.functional.silu(x)

    def forward(self, x: torch.Tensor):
        return self.forward_torch(x)

class SiluAndMul(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # shape: [batch_size, seq_len, 2d] -> [batch_size, seq_len, d] or [num_tokens, 2d] -> [num_tokens, d]
    def forward(self, x: torch.Tensor):
        d = x.shape[-1] // 2
        # the first half is result of gate, and the second half is the result of up_proj. The output is the element-wise product of the first half and the silu of the second half.
        return torch.nn.functional.silu(x[..., d:]) * x[..., :d]



if __name__ == '__main__':
    x = torch.randn(1, 2, 10)
    silu = SiLu()
    output1 = silu.forward_nativate(x)
    output2 = silu.forward_torch(x)
    assert torch.allclose(output1, output2, atol=1e-6), "The outputs are not close enough!"
