import pandas as pd

from .hooks import make_hooks, ablate_head
from .metrics import mean_logit_diff, induction_score


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


def run_head_sweep(model, induction_examples):
    baseline = induction_score(model, induction_examples)

    rows = []

    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            hooks = [
                (
                    f"blocks.{layer}.attn.hook_z",
                    lambda z, hook, h=head: ablate_head(z, hook, h)
                )
            ]

            score = induction_score(
                model, induction_examples, hooks=hooks
            )

            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "baseline_logit_diff": baseline,
                    "ablated_logit_diff": score,
                    "drop_in_logit_diff": baseline - score,
                }
            )

            print(layer, head, baseline - score)
    return pd.DataFrame(rows)