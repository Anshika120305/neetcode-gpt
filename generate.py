import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # Do not alter the fixed code below — it ensures reproducible test output.
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        
        # Initialize the output string
        generated_text = ""
        
        for i in range(new_chars):
            # 1. Crop context to context_length if it exceeds it
            context_cond = context[:, -context_length:]
            
            # 2. Get predictions/logits from the model
            logits = model(context_cond)
            
            # 3. Focus only on the last time step's logits and convert to probabilities
            logits = logits[:, -1, :] # shape: (1, vocab_size)
            probs = torch.softmax(logits, dim=-1)
            
            generator.set_state(initial_state)
            
            # 4. Sample the next token index
            next_token = torch.multinomial(probs, num_samples=1, generator=generator) # shape: (1, 1)
            
            # 5. Append sampled token to context
            context = torch.cat((context, next_token), dim=1)
            
            # 6. Map token ID to character and append to generated result
            char_idx = next_token.item()
            generated_text += int_to_char[char_idx]

        return generated_text
