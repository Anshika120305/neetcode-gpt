import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
       z = np.dot(w, x) + b
       y_hat = 1.0/(1.0 + np.exp(-z))
       delta = (y_hat - y_true)* y_hat*(1- y_hat)
       dl_dw = np.round(delta*x, 5)
       dl_db = float(np.round(delta, 5))
       return dl_dw, dl_db