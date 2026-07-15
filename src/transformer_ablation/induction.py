import torch
import random
import json

from pathlib import Path
from transformer_ablation.config import InductionExample

# Generate a set of prompts for testing induction behavior in the model
def generate_induction_prompts(model, num_examples=100, seq_len=5):

    examples = []

    vocab_size = model.cfg.d_vocab

    for _ in range(num_examples):

        # Need at least A B
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")

        # Create the initial sequence
        tokens = torch.randint(
            0,
            vocab_size,
            (seq_len,)
        )

        # Pick a token to repeat.
        # Avoid the last position because it needs a next token.
        repeat_position = random.randint(
            0,
            seq_len - 2
        )

        repeated_token = tokens[repeat_position]

        answer_token = tokens[repeat_position + 1]

        prompt_tokens = torch.cat(
            [
                tokens,
                repeated_token.unsqueeze(0)
            ]
        )

        examples.append(
            InductionExample(
                prompt=model.tokenizer.decode(
                    prompt_tokens
                ),
                answer=model.tokenizer.decode(
                    answer_token.unsqueeze(0)
                ),
                repeat_position=repeat_position
            )
        )

    return examples

def generate_natural_prompts():
    examples = [
        InductionExample(
            prompt="The cat sat on the mat. The cat",
            answer=" sat",
            repeat_position=1
        ),
        InductionExample(
            prompt="The dog chased the ball. The dog",
            answer=" chased",
            repeat_position=1
        ),
        InductionExample(
            prompt="Alice went to school. Alice",
            answer=" went",
            repeat_position=0
        ),
        InductionExample(
            prompt="Paris is beautiful in spring. Paris",
            answer=" is",
            repeat_position=0
        ),
    ]
    return examples

def create_custom_induction_prompt(prompt, answer, repeat_position):

    return [
        InductionExample(
            prompt=prompt,
            answer=answer,
            repeat_position=repeat_position
        )
    ]

def load_induction_prompts(path):
    with open(path, "r") as f:
        data = json.load(f)

    return [
        InductionExample(
            prompt=item["prompt"],
            answer=item["answer"],
            repeat_position=item["repeat_position"]
        )
        for item in data
    ]