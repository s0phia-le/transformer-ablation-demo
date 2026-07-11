import torch


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
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(ex["tokens"], fwd_hooks=hooks)

            else:
                logits = model(ex["tokens"])

        final_logits = logits[0,-1]

        target = ex["target"]

        scores.append(final_logits[target].item())

    return sum(scores) / len(scores)

def induction_attention_score(model, examples):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    scores = torch.zeros(n_layers, n_heads)

    counts = torch.zeros(n_layers, n_heads)

    for ex in examples:
        tokens = ex["tokens"]

        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: "pattern" in name 
        )

        for layer in range(n_layers):
            pattern = cache[
                f"blocks.{layer}.attn.hook_pattern"
            ]

            # shape: batch, heads, query, key
            #pattern = pattern[0]

            final_position = tokens.shape[1] - 1
            previous_position = 0
            pattern = pattern[:, :, final_position, previous_position]

            # check if final token attends to the first token
            for head in range(n_heads):
                attention = pattern[0, :, final_position, previous_position]

                scores[layer] += attention 
                counts[layer, head] += 1

    return scores / counts