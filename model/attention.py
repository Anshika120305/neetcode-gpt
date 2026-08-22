import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key_proj = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_proj = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_proj = nn.Linear(embedding_dim, attention_dim, bias=False)
        
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        K = self.key_proj(embedded)
        Q = self.query_proj(embedded)
        V = self.value_proj(embedded)
        
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32))
        
        # 3. Apply causal mask
        seq_len = embedded.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=embedded.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 4. Apply softmax along the last dimension
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 5. Output calculation and rounding to 4 decimal places
        output = torch.matmul(attention_weights, V)
        return torch.round(output, decimals=4)
        