import torch
from torch import nn

class VocabParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, embedding_dim))

    def forward(self, input_ids: torch.Tensor):
        return self.weight[input_ids]

# Shape: (2, 10, 512) -> [batch_size, seq_len, embedding_dim]
embedding = VocabParallelEmbedding(1000, 512)
input_ids = torch.randint(0, 1000, (2, 10))
output = embedding(input_ids)
print(output.shape)
