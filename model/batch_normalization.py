import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        X = np.array(x, dtype=np.float64)
        g = np.array(gamma, dtype=np.float64)
        b = np.array(beta, dtype=np.float64)
        rm = np.array(running_mean, dtype=np.float64)
        rv = np.array(running_var, dtype=np.float64)
        
        if training:
            
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            
           
            x_hat = (X - mean) / np.sqrt(var + eps)
            
           
            rm = (1.0 - momentum) * rm + momentum * mean
            rv = (1.0 - momentum) * rv + momentum * var
        else:
            
            x_hat = (X - rm) / np.sqrt(rv + eps)
        
       
        y = g * x_hat + b
        
       
        y_out = np.round(y, 4).tolist()
        rm_out = np.round(rm, 4).tolist()
        rv_out = np.round(rv, 4).tolist()
        
        return y_out, rm_out, rv_out