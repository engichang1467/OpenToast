"""Token Channel Selection (TCS) for FFN. Training-free, inference-time.

Wraps a timm-style Mlp (fc1, act, fc2). For each forward, importance is
estimated from a random token subset (2%-20% of N, depth-adaptive) and the
lowest-importance input channels of FC1 (dim D) and FC2 (dim 4D) are dropped by
slicing the weight into a dense sub-matrix. See paper Sec 3.2 / Algorithm 1.
"""
import types
import torch
import torch.nn.functional as F


def channel_importance(x, sample_rate, use_attn=False, cls_attn=None,
                       lam_cls=2.0, lam_patch=1.0, cls_index=0):
    """Per-channel importance from sampled tokens.

    x: [B, N, C]. Returns [C].
    Eq.7 (magnitude, default): mean |activation| over sampled patch tokens.
    Eq.6 (CLS-guided, use_attn=True): lam_cls*|x_cls| + lam_patch*mean(A_cls,i*|x_i|).
    cls_attn: [B, N] attention weights CLS->token, required when use_attn.
    """
    B, N, C = x.shape
    s = max(1, int(round(sample_rate * N)))
    idx = torch.randperm(N, device=x.device)[:s]
    xs = x[:, idx, :].abs()                       # [B, s, C]

    if use_attn and cls_attn is not None:
        a = cls_attn[:, idx].unsqueeze(-1)        # [B, s, 1]
        patch = (a * xs).mean(dim=(0, 1))         # [C]
        cls = x[:, cls_index, :].abs().mean(dim=0)  # [C]
        return lam_cls * cls + lam_patch * patch
    return xs.mean(dim=(0, 1))                     # [C]


def _tcs_forward(self, x):
    C = x.shape[-1]
    if self.r_fc1 > 0:
        I = channel_importance(x, self.sample_rate, self.use_attn,
                               getattr(self, "_cls_attn", None))
        k1 = max(1, round((1.0 - self.r_fc1) * C))
        c1 = I.topk(k1).indices
        h = F.linear(x[..., c1], self.fc1.weight[:, c1], self.fc1.bias)
    else:
        h = self.fc1(x)
    h = self.act(h)
    if self.r_fc2 > 0:
        I2 = channel_importance(h, self.sample_rate, self.use_attn,
                               getattr(self, "_cls_attn", None))
        k2 = max(1, round((1.0 - self.r_fc2) * h.shape[-1]))
        c2 = I2.topk(k2).indices
        out = F.linear(h[..., c2], self.fc2.weight[:, c2], self.fc2.bias)
    else:
        out = self.fc2(h)
    return out


def _sample_rate(depth, L, lo=0.02, hi=0.20):
    if L <= 1:
        return hi
    return lo + (hi - lo) * depth / (L - 1)


def _is_mlp(m):
    return hasattr(m, "fc1") and hasattr(m, "fc2") and hasattr(m, "act")


def _is_attn(m):
    return hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads")


def _cls_attn_hook(target_mlp, cls_index=0):
    """Forward pre-hook on a block's attention: recompute CLS->token weights.

    timm's fused SDPA never materializes the attention matrix, so we recompute
    just the CLS query row: softmax((q_cls @ kᵀ) * scale), averaged over heads.
    Stored on the paired MLP as `_cls_attn` [B, N] before the MLP runs.
    """
    def hook(module, args, kwargs):
        x = args[0]                                   # [B, N, D] (pre-attn norm input)
        B, N, D = x.shape
        H, hd = module.num_heads, module.head_dim
        qkv = module.qkv(x).reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)                        # [B, H, N, hd]
        qn, kn = getattr(module, "q_norm", None), getattr(module, "k_norm", None)
        if qn is not None and getattr(qn, "weight", None) is not None:
            q, k = qn(q), kn(k)
        q_cls = q[:, :, cls_index:cls_index + 1, :] * module.scale  # [B, H, 1, hd]
        a = (q_cls @ k.transpose(-2, -1)).softmax(-1)  # [B, H, 1, N]
        target_mlp._cls_attn = a.mean(1).squeeze(1)    # avg heads -> [B, N]
    return hook


def _attach_cls_hooks(model, cls_index=0):
    """Hook each transformer block's attention to feed CLS weights to its MLP."""
    for _, blk in model.named_modules():
        if hasattr(blk, "attn") and hasattr(blk, "mlp") \
                and _is_attn(blk.attn) and _is_mlp(blk.mlp):
            blk.attn.register_forward_pre_hook(_cls_attn_hook(blk.mlp, cls_index),
                                               with_kwargs=True)
    return model


def apply_tcs(model, fc1_ratios, fc2_ratios, use_attn=False,
              sample_lo=0.02, sample_hi=0.20, cls_index=0):
    """Attach TCS to every Mlp (in order). fc1_ratios/fc2_ratios length = #FFN.

    use_attn=True (CLS-distilled, e.g. DeiT) also hooks each block's attention
    to capture CLS->token weights for the Eq.6 attention-guided importance.
    """
    mlps = [m for _, m in model.named_modules() if _is_mlp(m)]
    assert len(mlps) == len(fc1_ratios) == len(fc2_ratios), \
        f"config length {len(fc1_ratios)} != model FFN count {len(mlps)}"
    L = len(mlps)
    for i, m in enumerate(mlps):
        m.r_fc1 = float(fc1_ratios[i])
        m.r_fc2 = float(fc2_ratios[i])
        m.use_attn = use_attn
        m.sample_rate = _sample_rate(i, L, sample_lo, sample_hi)
        m.forward = types.MethodType(_tcs_forward, m)
    if use_attn:
        _attach_cls_hooks(model, cls_index)
    return model
