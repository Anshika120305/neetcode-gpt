import numpy as np
from typing import List

class Solution:
    def forward_and_backward(
        self, 
        x: List[float], 
        W1: List[List[float]], 
        b1: List[float], 
        W2: List[List[float]], 
        b2: List[float], 
        y_true: List[float]
    ) -> dict:
        
        # Convert inputs to NumPy arrays
        x_np = np.array(x, dtype=np.float64)
        W1_np = np.array(W1, dtype=np.float64)
        b1_np = np.array(b1, dtype=np.float64)
        W2_np = np.array(W2, dtype=np.float64)
        b2_np = np.array(b2, dtype=np.float64)
        y_true_np = np.array(y_true, dtype=np.float64)
        
        # Forward pass
        z1 = np.dot(W1_np, x_np) + b1_np
        a1 = np.maximum(0, z1)
        z2 = np.dot(W2_np, a1) + b2_np
        
        # MSE Loss
        n = len(y_true_np)
        loss = np.mean((z2 - y_true_np) ** 2)
        
        # Backward pass
        dz2 = (2.0 / n) * (z2 - y_true_np)
        dW2 = np.outer(dz2, a1)
        db2 = dz2.copy()
        
        da1 = np.dot(W2_np.T, dz2)
        dz1 = da1 * (z1 > 0).astype(np.float64)
        dW1 = np.outer(dz1, x_np)
        db1 = dz1.copy()
        
        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }