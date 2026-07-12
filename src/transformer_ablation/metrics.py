import torch
import pandas as pd


def logit_diff_from_logits(logits, correct_id: int, incorrect_id: int) -> float:
    final_logits = logits[0, -1, :]
    return float((final_logits[correct_id] - final_logits[incorrect_id]).item())


def score_example(model, example, hooks=None) -> float:
    tokens = model.to_tokens(example.prompt)
    with torch.no_grad():
        if hooks is None:
            logits = model(tokens)
        else:
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    return logit_diff_from_logits(logits, example.correct_id, example.incorrect_id)


def mean_logit_diff(model, examples, hooks=None) -> float:
    scores = [score_example(model, ex, hooks=hooks) for ex in examples]
    return sum(scores) / len(scores)


def topk_predictions(model, prompt: str, hooks=None, k: int = 8) -> list[tuple[str, float]]:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        if hooks is None:
            logits = model(tokens)
        else:
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    top_probs, top_ids = torch.topk(probs, k)
    return [
        (model.tokenizer.decode([token_id]), float(prob))
        for token_id, prob in zip(top_ids.tolist(), top_probs.tolist())
    ]


def generate_continuation(model, prompt: str, hooks=None, max_new_tokens: int = 8) -> str:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        if hooks is None:
            output = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False, verbose=False)
        else:
            with model.hooks(fwd_hooks=hooks):
                output = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False, verbose=False)
    return model.tokenizer.decode(output[0, tokens.shape[1]:])

def induction_score(model, examples, hooks=None):

    scores = []

    for ex in examples:

        tokens = model.to_tokens(ex.prompt)

        answer_token = model.to_tokens(ex.answer, prepend_bos=False)

        answer_id = answer_token.item()

        with torch.no_grad():

            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)

        final_logits = logits[0, -1]

        score = final_logits[answer_id]

        scores.append(score.item())

    return sum(scores) / len(scores)

def induction_attention_score(model, examples, max_layers=None, max_heads=None):

    if max_layers is None:
        max_layers = model.cfg.n_layers

    if max_heads is None:
        max_heads = model.cfg.n_heads

    scores = {}

    for layer in range(max_layers):
        scores[layer] = torch.zeros(max_heads)

    for example in examples:

        tokens = model.to_tokens(example.prompt)

        _, cache = model.run_with_cache(tokens)

        for layer in range(max_layers):

            pattern = cache[f"blocks.{layer}.attn.hook_pattern"]

            query_position = tokens.shape[1] - 1 # final position
            key_position = example.repeat_position # position of the repeated token

            values = pattern[0, :max_heads, query_position, key_position]

            scores[layer] += values

    rows = []

    for layer in scores:
        for head in range(max_heads):
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "attention_score": float(
                        scores[layer][head] / len(examples)
                    )
                }
            )

    return pd.DataFrame(rows)