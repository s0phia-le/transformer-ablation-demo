# Transformer Ablation Study

A small mechanistic interpretability project for testing how GPT-2 Small changes when you ablate transformer layers, MLP blocks, and residual stream activations.

The metric is the mean logit difference:

```text
logit(correct next token) - logit(incorrect next token)
```

A larger `drop_in_logit_diff` means the ablated component mattered more for the prompts.

## Structure

```text
transformer_ablation_module/
├── configs/
│   └── default.yaml
├── data/
│   └── prompts.json
│   └── induction.json
│   └── induction_prompts.json
├── scripts/
│   └── run_ablation.py
├── src/
│   └── transformer_ablation/
│       ├── cli.py
│       ├── config.py
│       ├── experiment.py
│       ├── hooks.py
│       ├── induction.py
│       ├── metrics.py
│       ├── model.py
│       ├── plotting.py
│       └── prompts.py
├── results/
├── figures/
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install as an editable package:

```bash
pip install -e .
```

## Run

From the project root:

```bash
PYTHONPATH=src python scripts/run_ablation.py --config configs/default.yaml
```

If installed with `pip install -e .`, you can also run:

```bash
transformer-ablation --config configs/default.yaml
```

## Interactive demo

From the project root:

```bash
PYTHONPATH=src streamlit run scripts/app.py
```

Pick a prompt, an ablation type, and a layer in the sidebar to see the model's top next-token
predictions, generated continuation, and (for the built-in prompts) the correct-vs-incorrect
logit difference shift side-by-side with the unablated baseline. A "Run full sweep" button
reproduces the same layer-by-layer sweep as `scripts/run_ablation.py`, with an interactive chart.

## Outputs

```text
results/ablation_results.csv
figures/ablation_plot.png
```

## Ablations

### `whole_layer`

Zeros both attention output and MLP output for a transformer block:

```text
attn_out(layer) = 0
mlp_out(layer) = 0
```

### `mlp_only`

Zeros only the MLP output at a layer:

```text
mlp_out(layer) = 0
```

### `residual_stream`

Zeros the residual stream at the final token position before a layer:

```text
resid_pre(layer, final_position) = 0
```

## Edit prompts

Add or change examples in:

```text
data/prompts.json
```

Each correct and incorrect answer should be one token under the model tokenizer. The script skips examples where this is not true.

