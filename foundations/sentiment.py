import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x: TensorType[int]) -> TensorType[float]:
        out = self.embedding(x)
        
        # 2. Average across sequence length (dim=1): (B, T, 16) -> (B, 16)
        out = torch.mean(out, dim=1)
        
        # 3. Linear layer: (B, 16) -> (B, 1)
        out = self.linear(out)
        
        # 4. Sigmoid activation: (B, 1) -> (B, 1)
        out = self.sigmoid(out)
        
        # 5. Round output to 4 decimal places
        return torch.round(out, decimals=4)