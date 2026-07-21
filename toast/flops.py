"""Eager FLOPs counter for the projection/FFN matmuls.

fvcore jit-traces the model, which breaks on TCS's data-dependent channel
indexing (topk / randperm). Instead we patch F.linear during one eager forward
and sum 2*in*out*tokens using the ACTUAL runtime shapes -- this correctly
captures both SCWP-shrunk projections and TCS-sliced FFN sub-matrices (which
call F.linear manually, bypassing the nn.Linear module).

Counts linear/projection GEMMs only (the dominant, compressed cost). Excludes
attention QKᵀ/AV batched matmuls (~7-8% of FLOPs, uncompressed by ToaSt).
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def count_linear_gflops(model, x):
    total = 0
    orig = F.linear

    def patched(inp, weight, bias=None):
        nonlocal total
        tokens = inp.numel() // inp.shape[-1]
        in_f, out_f = inp.shape[-1], weight.shape[0]
        total += 2 * in_f * out_f * tokens
        return orig(inp, weight, bias)

    F.linear = patched
    try:
        model(x)
    finally:
        F.linear = orig
    return total / 1e9
