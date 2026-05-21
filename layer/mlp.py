# A naive torch implementation of MLP, which is not used in the final model. It is just for reference.
# Take Qwen3 dense model as an example, the MLP consists of two linear layers and one activation function in between. 
# The first linear layer expands the dimension from 512 to 2048, and the second linear layer reduces the dimension back to 512. 
# The activation function is usually SiLU.

from torch import nn
import torch.nn.functional as F
import torch


class SiluAndMul(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # shape: [batch_size, seq_len, 2d] -> [batch_size, seq_len, d] or [num_tokens, 2d] -> [num_tokens, d]
    def forward(self, x: torch.Tensor):
        d = x.shape[-1] // 2
        # the first half is result of gate, and the second half is the result of up_proj. The output is the element-wise product of the first half and the silu of the second half.
        # silu = x * sigmoid(x) = x * (1 / (1 + e^(-x))) = x / (1 + e^(-x))
        return torch.nn.functional.silu(x[..., :d]) * x[..., d:]

class LinearBase(nn.Module):
    def __init__(self, in_features, out_features, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, input: torch.Tensor):
        return F.linear(input, self.weight, self.bias)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = LinearBase(hidden_size, 2 * intermediate_size)
        self.down_proj = LinearBase(intermediate_size, hidden_size)
        self.act_fn = SiluAndMul()

    # output = down_proj( silu(gate_proj(x)) * up_proj(x) )
    def forward(self, x: torch.Tensor):
        assert x.shape[-1] == self.hidden_size, f"Input hidden size {x.shape[-1]} does not match expected {self.hidden_size}"
        gate_up_proj_output = self.gate_up_proj(x)  # [batch_size, seq_len, 2 * intermediate_size]
        act_output = self.act_fn(gate_up_proj_output)
        return self.down_proj(act_output)


if __name__ == '__main__':
    input = torch.randn(2, 10, 512)  # [batch_size, seq_len, hidden_size]
    mlp = MLP(hidden_size=512, intermediate_size=2048)
    output = mlp(input)
    print(output.shape)  # [batch_size, seq_len, hidden_size]
