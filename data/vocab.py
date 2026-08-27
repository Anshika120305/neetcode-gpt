from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        vocab = sorted(set(text))
        
        # Create mappings
        stoi = {ch: i for i, ch in enumerate(vocab)}
        itos = {i: ch for i, ch in enumerate(vocab)}
        return stoi, itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        return [stoi[ch]for ch in text]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        return "".join([itos[i]for i in ids])