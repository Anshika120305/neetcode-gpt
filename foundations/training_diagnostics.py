import torch
import torch.nn as nn
from typing import List, Dict

class Solution:
    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        """
        Forward pass through the model while recording output activation stats 
        for each nn.Linear layer.
        """
        stats = []
        current_input = x
        
        with torch.no_grad():
            for layer in model:
                current_input = layer(current_input)
                
                if isinstance(layer, nn.Linear):
                    # activations shape: (batch_size, num_neurons)
                    act = current_input
                    
                    mean_val = float(act.mean().item())
                    std_val = float(act.std().item())
                    
                    # A neuron is dead if it is <= 0 for ALL samples in the batch
                    # Check across batch dimension (dim=0)
                    is_dead_per_neuron = (act <= 0).all(dim=0)
                    dead_fraction = float(is_dead_per_neuron.float().mean().item())
                    
                    stats.append({
                        "mean": round(mean_val, 4),
                        "std": round(std_val, 4),
                        "dead_fraction": round(dead_fraction, 4)
                    })
                    
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        """
        Forward + backward pass using nn.MSELoss and compute weight gradient stats
        for each nn.Linear layer.
        """
        model.zero_grad()
        
        # Forward pass
        predictions = model(x)
        
        # MSE Loss & Backward pass
        criterion = nn.MSELoss()
        loss = criterion(predictions, y)
        loss.backward()
        
        stats = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                weight_grad = layer.weight.grad
                
                mean_val = float(weight_grad.mean().item())
                std_val = float(weight_grad.std().item())
                norm_val = float(torch.norm(weight_grad).item())
                
                stats.append({
                    "mean": round(mean_val, 4),
                    "std": round(std_val, 4),
                    "norm": round(norm_val, 4)
                })
                
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        """
        Diagnose the health of the neural network in priority order.
        """
        # 1. 'dead_neurons' if any layer has dead_fraction > 0.5
        for stat in activation_stats:
            if stat['dead_fraction'] > 0.5:
                return 'dead_neurons'
                
        # 2. 'exploding_gradients' if any layer gradient norm > 1000
        for stat in gradient_stats:
            if stat['norm'] > 1000:
                return 'exploding_gradients'
                
        # 3. 'vanishing_gradients' if last layer gradient norm < 1e-5
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
            
        # 4. Check activation std for all layers:
        #    if any std < 0.1 -> 'vanishing_gradients'
        #    if any std > 10.0 -> 'exploding_gradients'
        for stat in activation_stats:
            if stat['std'] < 0.1:
                return 'vanishing_gradients'
            if stat['std'] > 10.0:
                return 'exploding_gradients'
                
        # 5. 'healthy' if none of the above
        return 'healthy'