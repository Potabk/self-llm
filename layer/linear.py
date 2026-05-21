from torch import nn
import torch
import torch.nn.functional as F


class LinearBase(nn.Module):
    def __init__(self, in_features, out_features, device, bias: bool = False):
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.zeros(out_features), device=self.device) if bias else None
        self.weight = nn.Parameter(torch.randn(out_features, in_features), device=self.device)

    def forward(self, input: torch.Tensor):
        return F.linear(input, self.weight, self.bias)


if __name__ == '__main__':
    input = torch.randn(2, 10, 512)  # [batch_size, seq_len, embedding_dim]
    linear = LinearBase(512, 256)
    output = linear(input)
    print(output.shape)  # [batch_size, seq_len, out_features]
