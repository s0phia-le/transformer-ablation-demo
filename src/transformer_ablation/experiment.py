import pandas as pd

from .hooks import make_hooks, ablate_head
from .metrics import mean_logit_diff, induction_score, induction_attention_score


def run_ablation_sweep(model, examples, ablation_types: list[str]) -> pd.DataFrame:
    baseline = mean_logit_diff(model, examples)
    rows = []

    for layer in range(model.cfg.n_layers):
        for ablation_type in ablation_types:
            hooks = make_hooks(ablation_type, layer)
            ablated = mean_logit_diff(model, examples, hooks=hooks)
            drop = baseline - ablated
    
            rows.append({
                "layer": layer,
                "ablation_type": ablation_type,
                "baseline_logit_diff": baseline,
                "ablated_logit_diff": ablated,
                "drop_in_logit_diff": drop,
            })

            print(
                f"layer={layer:02d} type={ablation_type:16s} "
                f"ablated={ablated: .4f} drop={drop: .4f}"
            )

    return pd.DataFrame(rows)


def run_head_sweep(model, examples, max_layers=None,max_heads=None, progress=None, stop_flag=None):

    if max_layers is None:
        max_layers = model.cfg.n_layers

    if max_heads is None:
        max_heads = model.cfg.n_heads

    baseline = induction_score(
        model,
        examples
    )

    rows=[]
    completed = 0
    total = max_layers * max_heads

    for layer in range(max_layers):

        for head in range(max_heads):

            if stop_flag and stop_flag():
                print("Stopping head sweep")
                return pd.DataFrame(rows)

            hooks = [
                (
                    "blocks.%d.attn.hook_z" % layer,
                    lambda z, hook, h=head:
                        ablate_head(z, hook, h)
                )
            ]

            score = induction_score(
                model,
                examples,
                hooks=hooks
            )

            drop = baseline - score

            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "drop": drop
                }
            )

            completed += 1
            if progress:
                progress(completed / total)

    return pd.DataFrame(rows)

def run_attention_sweep(model, examples, max_layers=None, max_heads=None, progress=None, stop_flag=None):

    if max_layers is None:
        max_layers = model.cfg.n_layers

    if max_heads is None:
        max_heads = model.cfg.n_heads

    if stop_flag and stop_flag():
        return pd.DataFrame()

    attention_df = induction_attention_score(
        model,
        examples,
        max_layers=max_layers,
        max_heads=max_heads
    )

    return attention_df