from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch

from transformer_ablation.config import load_config
from transformer_ablation.diagram import architecture_diagram_svg
from transformer_ablation.experiment import run_ablation_sweep, run_head_sweep
from transformer_ablation.hooks import make_hooks
from transformer_ablation.metrics import generate_continuation, logit_diff_from_logits, topk_predictions
from transformer_ablation.model import load_model
from transformer_ablation.plotting import plot_layer_sweep
from transformer_ablation.prompts import build_examples
from transformer_ablation.induction import generate_induction_prompts

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
ABLATION_LABELS = {
    "none": "None (baseline)",
    "whole_layer": "Whole layer (attn + MLP out)",
    "mlp_only": "MLP output only",
    "residual_stream": "Residual stream (final token)",
}

PINK = "#d97ba6"
GOLD = "#c9992e"

st.set_page_config(page_title="Transformer Ablation Demo", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&display=swap');
    html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]) {
        font-family: 'Quicksand', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading GPT-2 Small...")
def get_model_and_examples():
    cfg = load_config(CONFIG_PATH)
    model = load_model(cfg.model_name, cfg.device)
    examples = build_examples(model, cfg.prompt_path, cfg.max_examples)
    return model, examples, cfg


def predictions_chart(pairs):
    df = pd.DataFrame(pairs, columns=["token", "probability"])
    return (
        alt.Chart(df)
        .mark_bar(color=GOLD)
        .encode(
            x=alt.X("token", sort=list(df["token"]), title=None),
            y=alt.Y("probability", title="probability"),
        )
    )


model, examples, cfg = get_model_and_examples()
n_layers = model.cfg.n_layers

st.title("Transformer Ablation Demo")
tab1, tab2 = st.tabs(["Layer Ablation", "Induction Head Ablation"])

with tab1:
    st.caption(
    f"{cfg.model_name} — watch next-token predictions and generated text shift as you ablate "
    "layers, MLP blocks, or the residual stream."
    )

    st.sidebar.header("Controls")

    example_by_id = {ex.id: ex for ex in examples}
    prompt_choice = st.sidebar.selectbox(
        "Prompt",
        options=["Custom"] + list(example_by_id.keys()),
        format_func=lambda k: k if k == "Custom" else f"{k}: {example_by_id[k].prompt}",
    )

    selected_example = None
    if prompt_choice == "Custom":
        prompt_text = st.sidebar.text_area("Enter a prompt", value="The capital of France is")
    else:
        selected_example = example_by_id[prompt_choice]
        prompt_text = selected_example.prompt

    ablation_type = st.sidebar.radio(
        "Ablation type",
        options=list(ABLATION_LABELS.keys()),
        format_func=lambda k: ABLATION_LABELS[k],
    )
    layer = st.sidebar.slider("Layer", 0, n_layers - 1, 0, disabled=(ablation_type == "none"))
    top_k = st.sidebar.slider("Top-k tokens to show", 3, 15, 8)
    max_new_tokens = st.sidebar.slider("Tokens to generate", 1, 20, 8)

    hooks = None if ablation_type == "none" else make_hooks(ablation_type, layer)
    ablated_label = "Baseline" if ablation_type == "none" else f"{ABLATION_LABELS[ablation_type]} @ layer {layer}"

    st.subheader("Prompt")
    st.code(prompt_text, language=None)

    st.subheader("Where does this hit the network?")
    st.caption(
        "GPT-2 Small's residual stream runs bottom (embedding) to top (logits) through every layer's "
        "attention and MLP sublayers. Red marks whatever the current selection zeroes out."
    )
    st.markdown(architecture_diagram_svg(n_layers, layer, ablation_type), unsafe_allow_html=True)

    col_base, col_ablated = st.columns(2)

    with col_base:
        st.markdown("**Baseline (no ablation)**")
        base_preds = topk_predictions(model, prompt_text, hooks=None, k=top_k)
        st.altair_chart(predictions_chart(base_preds), use_container_width=True)
        st.markdown("Generated continuation:")
        st.code(generate_continuation(model, prompt_text, hooks=None, max_new_tokens=max_new_tokens), language=None)

    with col_ablated:
        st.markdown(f"**{ablated_label}**")
        ablated_preds = topk_predictions(model, prompt_text, hooks=hooks, k=top_k)
        st.altair_chart(predictions_chart(ablated_preds), use_container_width=True)
        st.markdown("Generated continuation:")
        st.code(
            generate_continuation(model, prompt_text, hooks=hooks, max_new_tokens=max_new_tokens),
            language=None,
        )

    if selected_example is not None:
        st.subheader("Logit difference: correct vs. incorrect answer")
        tokens = model.to_tokens(prompt_text)

        with torch.no_grad():
            base_logits = model(tokens)
            ablated_logits = base_logits if hooks is None else model.run_with_hooks(tokens, fwd_hooks=hooks)

        base_diff = logit_diff_from_logits(base_logits, selected_example.correct_id, selected_example.incorrect_id)
        ablated_diff = logit_diff_from_logits(
            ablated_logits, selected_example.correct_id, selected_example.incorrect_id
        )

        m1, m2 = st.columns(2)
        m1.metric(f"'{selected_example.correct.strip()}' minus '{selected_example.incorrect.strip()}' (baseline)", f"{base_diff:.3f}")
        m2.metric(
            f"Same, ablated",
            f"{ablated_diff:.3f}",
            delta=f"{ablated_diff - base_diff:.3f}",
            delta_color="inverse",
        )

    st.divider()
    st.subheader("Full layer sweep")
    st.write(
        "Runs every ablation type across every layer, averaged over all prompts in "
        f"`{cfg.prompt_path.name}` — the same sweep as `scripts/run_ablation.py`."
    )

    if st.button("Run full sweep", type="primary"):
        with st.spinner("Running sweep across all layers..."):
            st.session_state["sweep_df"] = run_ablation_sweep(model, examples, cfg.ablation_types)

    if "sweep_df" in st.session_state:
        df = st.session_state["sweep_df"]
        sweep_chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X("layer", title="Layer"),
                y=alt.Y("drop_in_logit_diff", title="Drop in logit difference"),
                color=alt.Color(
                    "ablation_type",
                    title="Ablation type",
                    scale=alt.Scale(range=[GOLD, PINK, "#c65a4a"]),
                ),
            )
        )
        st.altair_chart(sweep_chart, use_container_width=True)
        st.dataframe(df, use_container_width=True)

        cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.output_csv, index=False)
        plot_layer_sweep(df, cfg.output_plot)
        st.download_button("Download CSV", df.to_csv(index=False), file_name="ablation_results.csv")

with tab2:

    st.header("Induction Head Detection")

    if st.button("Find induction heads"):

        with st.spinner(
            "Testing all attention heads..."
        ):

            induction_examples = generate_induction_prompts(
                model,
                num_examples=200
            )

            df = run_head_sweep(
                model,
                induction_examples
            )
            st.session_state["head_df"] = df

    if "head_df" in st.session_state:

        df = st.session_state["head_df"]

        df = df.sort_values(
            "drop",
            ascending=False
        )

        st.dataframe(
            df.head(20)
        )

    chart = (
        alt.Chart(df.head(20))
        .mark_bar()
        .encode(
            x=alt.X("layer:N"),
            y=alt.Y("drop:Q"),
            color="head:N"
        )
    )
    st.altair_chart(chart)

