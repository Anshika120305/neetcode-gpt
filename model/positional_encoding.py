import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        pos = np.arange(seq_len)[:, np.newaxis]
        
        # Create dimension indices array for pairs (i = 0, 1, ..., d_model/2 - 1)
        i = np.arange(0, d_model, 2)
        
        # Calculate the division term: 10000 ^ (2i / d_model)
        div_term = 10000 ** (i / d_model)
        
        # Initialize output array of shape (seq_len, d_model)
        pe = np.zeros((seq_len, d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = np.sin(pos / div_term)
        pe[:, 1::2] = np.cos(pos / div_term)
        
        # Round values to 5 decimal places as specified
        return np.round(pe, 5)

        