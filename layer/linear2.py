from torch import nn
import torch.nn.functional as F
import torch


class Linear2(nn.Module):
    def __init__(self, in_features, out_features, bias: bool = False):  
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
    
    
    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias)


class VocabEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, embedding_dim))
    
    
    def forward(self, input_ids: torch.Tensor):
        return self.weight[input_ids]


if __name__ == '__main__':
    input = torch.randn(2, 10, 512)
    embedding = VocabEmbedding(1000, 512)
    embed_tokens = embedding(torch.randint(0, 10, (2, 10)))  # [batch_size, seq_len, embedding_dim]
    print(embed_tokens.shape)  # [batch_size, seq_len, embedding_dim]
    print(embed_tokens[0, 0, :].shape)
    print(embed_tokens[0, 0, :])
