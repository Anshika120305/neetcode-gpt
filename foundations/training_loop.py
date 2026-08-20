import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        n_samples, n_features = X.shape
        
        # Initialize weights to zeros and bias to 0
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        
        for _ in range(epochs):
            # Forward pass: predictions y_hat = X @ w + b
            y_hat = X @ w + b
            
            # Error difference (y_hat - y)
            error = y_hat - y
            
            # Gradient calculations:
            # dL/dw = (2 / n) * X^T @ (y_hat - y)
            dw = (2 / n_samples) * (X.T @ error)
            
            # dL/db = (2 / n) * sum(y_hat - y)
            db = (2 / n_samples) * np.sum(error)
            
            # Parameter updates
            w -= lr * dw
            b -= lr * db
            
        # Return weights and bias rounded to 5 decimal places
        return np.round(w, 5), round(b, 5)