import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
       z = np.dot(x,w) +b
       if activation == "sigmoid":
            val = 1 / (1 + np.exp(-z))
       elif activation == "relu":
            val = max(0.0, float(z))
            
       return round(float(val), 5)
       
       
    
