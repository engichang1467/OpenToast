"""Structured Coupled Weight Pruning (SCWP) for MHSA.

Offline pruning of the per-head dimension dk -> d'k via geometric-median
importance, with coupled index synchronization (Q-K share indices, V-Proj
share indices). See paper Sec 3.1 / Algorithm 1.

Works on timm-style fused attention: `attn.qkv` = Linear(D, 3D),
`attn.proj` = Linear(D, D), `attn.num_heads = H`. Layout of qkv.weight rows:
[Q(0:D) ; K(D:2D) ; V(2D:3D)], each block reshaped per head to (H, dk, D).
"""
import torch
import torch.nn as nn


def geometric_median(points, eps=1e-6, iters=200):
    """Weiszfeld geometric median of a set of points.

    points: [P, dim] (P vectors). Returns [dim] minimizing sum of L2 distances.
    """
    y = points.mean(dim=0)
    for _ in range(iters):
        d = torch.clamp((points - y).norm(dim=1), min=eps)
        w = 1.0 / d
        y_new = (w[:, None] * points).sum(dim=0) / w.sum()
        if (y_new - y).norm() < eps:
            break
        y = y_new
    return y


def _coupled_keep_indices(qh, kh, vh, projh, dpk):
    """Per-head kept indices via averaged coupled GM importance.

    qh, kh, vh: [dk, D] (row j = paper weight column j, length D).
    projh:      [D, dk]  (column j = paper W_proj row j, length D).
    Returns sorted LongTensor of length dpk (dims to keep = highest distance).
    """
    return _coupled_importance(qh, kh, vh, projh).topk(dpk).indices.sort().values


def _coupled_importance(qh, kh, vh, projh):
    """Averaged coupled GM distance per dk dimension. Returns [dk] (low=redundant)."""
    x_qk = torch.cat([qh, kh], dim=1)            # [dk, 2D]
    x_vo = torch.cat([vh, projh.t()], dim=1)     # [dk, 2D]
    i_qk = (x_qk - geometric_median(x_qk)).norm(dim=1)
    i_vo = (x_vo - geometric_median(x_vo)).norm(dim=1)
    return 0.5 * (i_qk + i_vo)


@torch.no_grad()
def prune_attention(attn, ratio):
    """Prune one timm attention module in place. ratio in (0,1)."""
    H = attn.num_heads
    W = attn.qkv.weight                          # [3D, D]
    D = W.shape[1]
    dk = D // H
    dpk = max(1, round((1.0 - ratio) * dk))
    b = attn.qkv.bias

    q, k, v = W[:D], W[D:2 * D], W[2 * D:]        # each [H*dk, D]
    q = q.reshape(H, dk, D); k = k.reshape(H, dk, D); v = v.reshape(H, dk, D)
    pw = attn.proj.weight                         # [D, H*dk]
    proj = pw.reshape(D, H, dk)                    # [D, H, dk]

    bq = bk = bv = None
    if b is not None:
        bq, bk, bv = b[:D].reshape(H, dk), b[D:2 * D].reshape(H, dk), b[2 * D:].reshape(H, dk)

    # q_norm/k_norm (timm qk_norm) are a single LayerNorm/RMSNorm shared over
    # head_dim across all heads. Per-head keep indices can't slice a shared
    # param vector consistently, so when qk_norm is active we select ONE keep
    # set (importance summed across heads) and reuse it for every head. This
    # keeps the head-wise-uniform count while letting us slice the norm once.
    imps = [_coupled_importance(q[h], k[h], v[h], proj[:, h, :]) for h in range(H)]
    has_qk_norm = _affine(getattr(attn, "q_norm", None)) or _affine(getattr(attn, "k_norm", None))
    shared_keep = torch.stack(imps).sum(0).topk(dpk).indices.sort().values if has_qk_norm else None

    nq, nk, nv, nproj = [], [], [], []
    nbq, nbk, nbv = [], [], []
    for h in range(H):
        keep = shared_keep if has_qk_norm else imps[h].topk(dpk).indices.sort().values
        nq.append(q[h][keep]); nk.append(k[h][keep]); nv.append(v[h][keep])
        nproj.append(proj[:, h, :][:, keep])      # [D, dpk]
        if b is not None:
            nbq.append(bq[h][keep]); nbk.append(bk[h][keep]); nbv.append(bv[h][keep])

    new_qkv_w = torch.cat([torch.cat(nq), torch.cat(nk), torch.cat(nv)], dim=0)  # [3*H*dpk, D]
    new_proj_w = torch.cat(nproj, dim=1)          # [D, H*dpk]

    new_qkv = nn.Linear(D, 3 * H * dpk, bias=b is not None).to(W.device, W.dtype)
    new_qkv.weight.copy_(new_qkv_w)
    if b is not None:
        new_qkv.bias.copy_(torch.cat([torch.cat(nbq), torch.cat(nbk), torch.cat(nbv)]))

    new_proj = nn.Linear(H * dpk, D, bias=attn.proj.bias is not None).to(W.device, W.dtype)
    new_proj.weight.copy_(new_proj_w)
    if attn.proj.bias is not None:
        new_proj.bias.copy_(attn.proj.bias)

    attn.qkv, attn.proj = new_qkv, new_proj
    if has_qk_norm:
        _slice_norm(getattr(attn, "q_norm", None), shared_keep, dpk)
        _slice_norm(getattr(attn, "k_norm", None), shared_keep, dpk)
    # timm reads self.head_dim (reshape) and self.attn_dim (output reshape).
    # Swin's WindowAttention infers head_dim via reshape(-1) and has no attn_dim.
    if hasattr(attn, "head_dim"):
        attn.head_dim = dpk
    if hasattr(attn, "attn_dim"):
        attn.attn_dim = H * dpk
    attn.scale = dpk ** -0.5


def _affine(norm):
    """True if norm is a real (learnable) LayerNorm/RMSNorm over head_dim."""
    return norm is not None and getattr(norm, "weight", None) is not None


def _slice_norm(norm, keep, dpk):
    """Slice a shared head_dim LayerNorm/RMSNorm to the kept indices."""
    if not _affine(norm):
        return
    norm.weight = nn.Parameter(norm.weight.data[keep].clone())
    if getattr(norm, "bias", None) is not None:
        norm.bias = nn.Parameter(norm.bias.data[keep].clone())
    if hasattr(norm, "normalized_shape"):
        norm.normalized_shape = (dpk,)


def _is_attn(m):
    return hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads")


@torch.no_grad()
def apply_scwp(model, ratio, skip_idx=(0,)):
    """Prune every attention module (in registration order), skipping skip_idx.

    skip_idx: global attention indices to leave intact (layer 1, and first
    block of each Swin stage).
    """
    attns = [m for _, m in model.named_modules() if _is_attn(m)]
    for i, a in enumerate(attns):
        if i in skip_idx:
            continue
        prune_attention(a, ratio)
    return model
