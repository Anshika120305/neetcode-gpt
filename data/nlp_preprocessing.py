import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        all_sentences = positive + negative
        
        # Collect all unique words
        vocab_set = set()
        for sentence in all_sentences:
            vocab_set.update(sentence.split())
            
        # Sort words lexicographically and map each word to a unique ID starting at 1
        vocab = {word: i + 1 for i, word in enumerate(sorted(vocab_set))}
        
        # Encode each sentence into a PyTorch tensor of float IDs
        encoded_tensors = []
        for sentence in all_sentences:
            ids = [float(vocab[word]) for word in sentence.split()]
            encoded_tensors.append(torch.tensor(ids, dtype=torch.float32))
            
        # Pad shorter sequences with 0s to create a rectangular tensor of shape (2N, T)
        padded_dataset = nn.utils.rnn.pad_sequence(encoded_tensors, batch_first=True, padding_value=0.0)
        
        return padded_dataset