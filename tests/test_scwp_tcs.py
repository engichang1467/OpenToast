"""Sanity checks for SCWP + TCS. Run: python -m pytest tests/ (or python this file)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from toast.scwp import geometric_median, prune_attention
from toast.tcs import apply_tcs, channel_importance


class TimmLikeAttention(nn.Module):
    """Minimal replica of timm ViT Attention (uses self.head_dim in forward)."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        a = (q @ k.transpose(-2, -1) * self.scale).softmax(-1)
        o = (a @ v).transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        return self.proj(o)


class TimmLikeMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_geometric_median():
    pts = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    gm = geometric_median(pts)
    assert gm.norm() < 1e-3  # median at origin


def test_scwp_shapes_and_interface():
    D, H, N = 64, 8, 16
    attn = TimmLikeAttention(D, H)
    x = torch.randn(2, N, D)
    y0 = attn(x)
    prune_attention(attn, ratio=0.5)          # dk 8 -> 4
    dpk = 4
    assert attn.qkv.out_features == 3 * H * dpk
    assert attn.proj.in_features == H * dpk
    assert attn.head_dim == dpk
    y1 = attn(x)
    assert y1.shape == y0.shape == (2, N, D)   # block interface (D) preserved


def test_tcs_reduces_and_preserves_interface():
    D, N = 32, 20
    mlp = TimmLikeMlp(D, 4 * D)
    # wrap a single mlp via the public API on a 1-module container
    holder = nn.Module(); holder.m = mlp
    apply_tcs(holder, fc1_ratios=[0.5], fc2_ratios=[0.9])
    x = torch.randn(2, N, D)
    out = mlp(x)
    assert out.shape == (2, N, D)              # D interface preserved


def test_channel_importance_shape():
    x = torch.randn(2, 20, 32)
    I = channel_importance(x, sample_rate=0.2)
    assert I.shape == (32,)
    assert (I >= 0).all()


if __name__ == "__main__":
    test_geometric_median()
    test_scwp_shapes_and_interface()
    test_tcs_reduces_and_preserves_interface()
    test_channel_importance_shape()
    print("all sanity checks passed")
