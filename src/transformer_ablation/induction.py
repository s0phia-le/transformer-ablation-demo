import torch
import random

def generate_induction_prompts(model, num_examples=100, seq_len=5):
    examples = []
    vocab_size = model.cfg.d_vocab

    for i in range(num_examples):
        # generate random tokens for the prompt
        tokens = torch.randint(0, vocab_size, (seq_len,))

        # ensure that the last token is repeated to create an induction pattern
        repeated_token = tokens[0]

        prompt_tokens = torch.cat(
            [
                tokens[1:],
                repeated_token.unsqueeze(0)
            ]
        )
        # the target is the next token in the sequence, which should be the repeated token
        target = tokens[1]

        examples.append(
            {
                "tokens": prompt_tokens.unsqueeze(0),
                "target": target
            }
        )
    return examples