import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        curr = x
        n_layers = len(weights)
        
        for i in range(n_layers):
            
            curr = np.matmul(curr, weights[i]) + biases[i]
            
            if i < n_layers - 1:
                curr = np.maximum(0, curr)
                
        return np.round(curr, 5)