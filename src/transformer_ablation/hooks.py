import torch
from transformer_lens import utils


def zero_hook(x, hook):
    return torch.zeros_like(x)


def zero_final_position_hook(x, hook):
    x = x.clone()
    x[:, -1, :] = 0
    return x


def whole_layer_hooks(layer: int):
    return [
        (utils.get_act_name("attn_out", layer), zero_hook),
        (utils.get_act_name("mlp_out", layer), zero_hook),
    ]


def mlp_only_hooks(layer: int):
    return [(utils.get_act_name("mlp_out", layer), zero_hook)]


def residual_stream_hooks(layer: int):
    return [(utils.get_act_name("resid_pre", layer), zero_final_position_hook)]


def make_hooks(ablation_type: str, layer: int):
    if ablation_type == "whole_layer":
        return whole_layer_hooks(layer)
    if ablation_type == "mlp_only":
        return mlp_only_hooks(layer)
    if ablation_type == "residual_stream":
        return residual_stream_hooks(layer)
    raise ValueError(f"Unknown ablation type: {ablation_type}")

# Hooks for ablation of individual attention heads
def ablate_head(z, hook, head):
    z = z.clone()
    z[:, :, head, :] = 0
    return z

# github.com/globalsecurepayments/anthropic-fellowship/tree/master/mech-interp
