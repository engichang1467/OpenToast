"""Sanity + integration checks for SCWP + TCS.

Run:  PYTHONPATH=. python tests/test_scwp_tcs.py
Real-model tests need timm; they self-skip if unavailable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from toast.scwp import geometric_median, prune_attention, apply_scwp, _coupled_keep_indices
from toast.tcs import apply_tcs, channel_importance
from toast.flops import count_linear_gflops


# ---------------------------------------------------------------- unit: SCWP

def test_geometric_median():
    pts = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    assert geometric_median(pts).norm() < 1e-3          # median at origin


def test_coupled_keep_drops_redundant_dims():
    # Build a head where dims 0,1 sit near the geometric median (redundant)
    # and the rest are pushed far out (distinctive). Keep must exclude 0,1.
    dk, D = 6, 8
    torch.manual_seed(0)
    q = torch.randn(dk, D); k = torch.randn(dk, D)
    v = torch.randn(dk, D); proj = torch.randn(D, dk)
    # collapse dims 0,1 toward the centroid of the others
    for t, near in ((q, q), (k, k), (v, v)):
        t[0] = t[2:].mean(0); t[1] = t[2:].mean(0)
    proj[:, 0] = proj[:, 2:].mean(1); proj[:, 1] = proj[:, 2:].mean(1)
    keep = _coupled_keep_indices(q, k, v, proj, dpk=4).tolist()
    assert 0 not in keep and 1 not in keep


class TimmLikeAttention(nn.Module):
    """Replica of timm ViT Attention (uses head_dim + attn_dim in forward)."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dim = dim
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        a = (q @ k.transpose(-2, -1) * self.scale).softmax(-1)
        o = (a @ v).transpose(1, 2).reshape(B, N, self.attn_dim)
        return self.proj(o)


def test_scwp_shapes_and_interface():
    D, H, N = 64, 8, 16
    attn = TimmLikeAttention(D, H)
    x = torch.randn(2, N, D)
    y0 = attn(x)
    prune_attention(attn, ratio=0.5)                    # dk 8 -> 4
    dpk = 4
    assert attn.qkv.out_features == 3 * H * dpk
    assert attn.proj.in_features == H * dpk
    assert attn.head_dim == dpk and attn.attn_dim == H * dpk
    y1 = attn(x)
    assert y1.shape == y0.shape == (2, N, D)            # block interface preserved


class TimmLikeAttentionQKNorm(TimmLikeAttention):
    """timm Attention with qk_norm: shared LayerNorm(head_dim) on q and k."""
    def __init__(self, dim, num_heads):
        super().__init__(dim, num_heads)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)          # normalize over head_dim
        a = (q @ k.transpose(-2, -1) * self.scale).softmax(-1)
        o = (a @ v).transpose(1, 2).reshape(B, N, self.attn_dim)
        return self.proj(o)


def test_scwp_qk_norm_sliced():
    D, H, N = 64, 8, 16
    attn = TimmLikeAttentionQKNorm(D, H)
    x = torch.randn(2, N, D)
    y0 = attn(x)
    prune_attention(attn, ratio=0.5)                    # dk 8 -> 4
    dpk = 4
    # norm params sliced to dpk so the module runs without a shape error
    assert attn.q_norm.weight.shape == (dpk,) and attn.k_norm.bias.shape == (dpk,)
    assert attn.q_norm.normalized_shape == (dpk,)
    y1 = attn(x)
    assert y1.shape == y0.shape == (2, N, D)


# ---------------------------------------------------------------- unit: TCS

class TimmLikeMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_tcs_preserves_interface_and_slices():
    D, N = 32, 20
    mlp = TimmLikeMlp(D, 4 * D)
    holder = nn.Module(); holder.m = mlp
    apply_tcs(holder, fc1_ratios=[0.5], fc2_ratios=[0.9])
    x = torch.randn(2, N, D)
    # count matmul FLOPs of the wrapped mlp; must be < dense mlp
    dense = TimmLikeMlp(D, 4 * D)
    assert mlp(x).shape == (2, N, D)                     # D interface preserved
    assert count_linear_gflops(mlp, x) < count_linear_gflops(dense, x)


def test_channel_importance_shape_and_topk():
    x = torch.randn(2, 20, 32)
    I = channel_importance(x, sample_rate=0.2)
    assert I.shape == (32,) and (I >= 0).all()
    k = 10
    assert I.topk(k).indices.shape == (k,)


def test_channel_importance_cls_weighting():
    # CLS-guided (Eq.6) must differ from magnitude (Eq.7) when attn is provided
    x = torch.randn(2, 20, 16)
    cls_attn = torch.rand(2, 20)
    mag = channel_importance(x, 1.0)
    cls = channel_importance(x, 1.0, use_attn=True, cls_attn=cls_attn)
    assert not torch.allclose(mag, cls)


# ---------------------------------------------------- integration: real timm

def _timm():
    try:
        import timm
        return timm
    except ImportError:
        return None


def test_real_deit_small_end_to_end():
    timm = _timm()
    if timm is None:
        print("SKIP real deit (no timm)"); return
    m = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=1000).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        g_base = count_linear_gflops(m, x)
        y0 = m(x)
    apply_scwp(m, 0.90, (0,))
    fc1 = [0] * 10 + [0.5, 0.5]
    fc2 = [0] * 7 + [0.8, 0.8, 0.9, 0.9, 0.9]
    apply_tcs(m, fc1, fc2, use_attn=False)
    with torch.no_grad():
        y1 = m(x)
        g_toast = count_linear_gflops(m, x)
    assert y1.shape == y0.shape == (2, 1000)
    assert g_toast < 0.75 * g_base                       # meaningful FLOPs cut
    print(f"deit-small linear GFLOPs (x1): {g_base/2:.2f} -> {g_toast/2:.2f}")


def test_real_swin_tiny_scwp():
    timm = _timm()
    if timm is None:
        print("SKIP real swin (no timm)"); return
    m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=1000).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y0 = m(x)
    apply_scwp(m, 0.90, (0, 2, 4, 10))                   # skip first block of each stage
    with torch.no_grad():
        y1 = m(x)
    assert y1.shape == y0.shape == (2, 1000)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("\nall checks passed")
