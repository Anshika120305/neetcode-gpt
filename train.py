import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            # 1. Set seed for reproducible batch sampling per epoch
            torch.manual_seed(epoch)
            
            # 2. Sample random start indices
            # Valid start indices range from 0 up to len(data) - context_length - 1
            max_idx = len(data) - context_length
            ix = torch.randint(0, max_idx, (batch_size,))
            
            # 3. Build X and Y batches
            x = torch.stack([data[i:i + context_length] for i in ix])
            y = torch.stack([data[i + 1:i + context_length + 1] for i in ix])
            
            # 4. Forward pass
            logits = model(x)  # Shape: (batch_size, context_length, vocab_size)
            
            # 5. Reshape for cross-entropy loss
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = y.view(B * T)
            
            # 6. Compute loss
            loss = F.cross_entropy(logits_flat, targets_flat)
            
            # 7. Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return round(loss.item(), 4)