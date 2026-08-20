import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x_arr = np.array(x, dtype=np.float64)
        gamma_arr = np.array(gamma, dtype=np.float64)
        rms = np.sqrt(np.mean(x_arr ** 2) + eps)
        x_norm = (x_arr / rms) * gamma_arr
        return [round(float(val), 4) for val in x_norm]