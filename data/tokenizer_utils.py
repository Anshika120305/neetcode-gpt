from typing import List, Dict

class Solution:
    def _greedy_tokenize(self, s: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        n = len(s)
        
        while i < n:
            match = None
            # Find the longest matching prefix in vocab starting at index i
            for end in range(n, i, -1):
                sub = s[i:end]
                if sub in vocab:
                    match = sub
                    break
            
            if match:
                tokens.append(match)
                i += len(match)
            else:
                # If no multi-char or single-char vocab match exists, consume single character
                tokens.append(s[i])
                i += 1
                
        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        return [self._greedy_tokenize(str(num), vocab) for num in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        return len(self._greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        words = text.split()
        if not words:
            return 0.0
        
        total_tokens = self.count_tokens(text, vocab)
        return round(total_tokens / len(words), 4)
