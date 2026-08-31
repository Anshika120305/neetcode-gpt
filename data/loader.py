import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        torch.manual_seed(0)
        
        # Max valid starting index ensures target slice data[start + 1 : start + 1 + context_length] doesn't exceed bounds
        max_start = len(data) - context_length
        ix = torch.randint(0, max_start, (batch_size,))
        
        X = torch.stack([data[i : i + context_length] for i in ix])
        Y = torch.stack([data[i + 1 : i + 1 + context_length] for i in ix])
        
        return X, Y