from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
      from collections import Counter
from typing import List


class Solution:

    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        if not corpus:
            return []

        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            if len(tokens) < 2:
                break

            # Count frequency of all adjacent token pairs
            pair_counts = Counter(
                zip(tokens[:-1], tokens[1:])
            )

            if not pair_counts:
                break

            # Find the maximum frequency
            max_freq = max(pair_counts.values())

            # Filter pairs with maximum frequency and break ties lexicographically
            most_frequent_pair = min(
                pair for pair, freq in pair_counts.items() if freq == max_freq
            )

            token_a, token_b = most_frequent_pair
            merges.append([token_a, token_b])

            # Merge occurrences left-to-right without overlapping
            merged_token = token_a + token_b
            new_tokens = []
            i = 0
            n = len(tokens)

            while i < n:
                if (
                    i < n - 1
                    and tokens[i] == token_a
                    and tokens[i + 1] == token_b
                ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return merges