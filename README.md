# ToaSt: Token Channel Selection and Structured Pruning for Efficient ViT

Reimplementation of **ToaSt** (Moon, Park, Waslander — ICML 2026, PMLR 306).
Paper source: arXiv:2602.15720. Official code: `github.com/SHANNonLab-HUFS/ToaSt`.

ToaSt is a **decoupled, layer-independent** ViT compression framework. Two
independent components, applied to different parts of each transformer block:

1. **SCWP — Structured Coupled Weight Pruning** for MHSA. Offline. Prunes the
   per-head dimension `dk`; needs fine-tuning to recover.
2. **TCS — Token Channel Selection** for FFN. Online / training-free. Selects
   channels at inference time; no retraining.

Both target the **channel dimension `D`**, leaving the token sequence `N`
intact — so ToaSt is orthogonal to (composes with) token-merging methods like
ToMe.

---

## 1. Method summary (what to implement)

### Prerequisites / notation
- ViT with `L` layers, input `X ∈ R^{N×D}`, `H` heads, `dk = D/H`.
- MHSA weights per head: `W_Q, W_K, W_V ∈ R^{D×dk}`, `W_proj ∈ R^{dk×D}`.
- FFN: `W_FC1 ∈ R^{D×4D}`, `W_FC2 ∈ R^{4D×D}`, GELU between them.
- Block: `X' = MHSA(LN(X)) + X`, `X = FFN(LN(X')) + X'`.

### 1a. SCWP — MHSA pruning (offline, then fine-tune)
Reduce `dk → d'k = ⌊(1−ρm)·dk⌋`, uniform across all heads.

**Coupled synchronization (mandatory — non-aligned pruning collapses accuracy):**
- **Q-K:** prune column `j` of `W_Q` ⇒ prune column `j` of `W_K` (preserves dot product).
- **V-Proj:** prune column `j` of `W_V` ⇒ prune row `j` of `W_proj`.

**Importance = distance from Geometric Median (GM):**
- Form coupled matrices `W_QK = [W_Q; W_K]`, `W_VO = [W_V; W_projᵀ]`.
- `I_QK[j] = ‖w_QK,j − GM(W_QK)‖₂`, `I_VO[j] = ‖w_VO,j − GM(W_VO)‖₂`.
- Keep indices: `TopK( ½(I_QK + I_VO), d'k )` per head.
- GM ≈ L1 ≈ L2 (gap ≤0.66%); GM slightly best. Implement GM via `He et al. 2019` (FPGM).

**Schedule:** skip layer 1 (preserve embedding interface); prune the rest.
`ρm = 0.90` (0.80 for tiny models). For Swin, skip the **first layer of each stage**.
Then fine-tune (AdamW + cosine LR). Head-wise uniform keeps GEMM batched (no padding).

### 1b. TCS — FFN pruning (online, training-free)
For each block, prune FC1 and FC2 **independently** on their input channel axis.

**Statistical token sampling:** compute importance over a random subset
`S ⊂ {1..N}`, `|S| ∈ [0.02N, 0.2N]` (sample rate grows with layer depth).
Justified by high linear channel dependency (R² ≈ 1.0) — even 1 token keeps >97% acc.

**Importance score per channel `c`:**
- CLS-distilled models (DeiT):
  `I_c = λ_cls·|x_cls^(c)| + λ_patch·(1/|S|)·Σ_{i∈S} A_cls,i·|x_i^(c)|`,
  with `λ_cls = 2.0`, `λ_patch = 1.0` (stable for `λ_cls ∈ [1,3]`).
- Non-CLS models (ViT-MAE, Swin): `I_c = (1/|S|)·Σ_{i∈S} |x_i^(c)|` (magnitude only).
  Attention weighting is neutral here — architecture-dependent switch.

**Selection & reduction:** keep `TopK(I, k)`; `k1 = ⌊(1−r_FC1)·D⌋`, `k2 = ⌊(1−r_FC2)·4D⌋`.
Slice input columns of the weight → dense sub-matrix → standard GEMM (no sparse kernels).

**Layer-adaptive ratios ("fc2-heavy"):** conservative FC1 (0–30%), aggressive FC2
(50–90%) in deeper layers. This pattern beats uniform/stepwise/inverse across
architectures (inverse collapses to 18% on DeiT-S).

### Full pipeline (Algorithm 1)
```
# Offline: SCWP
for layer l in 1..L:
  for head h in 1..H:
    W_QK = [W_Q; W_K];  W_VO = [W_V; W_projᵀ]
    I_QK[j] = ‖w_QK,j − GM(W_QK)‖₂;  I_VO[j] = ‖w_VO,j − GM(W_VO)‖₂
    K_h = TopK(½(I_QK+I_VO), d'k)          # d'k = ⌊(1−ρm)dk⌋, uniform
  gather cols of W_Q,W_K,W_V and rows of W_proj at K_h   # dk→d'k
# (fine-tune the pruned MHSA here)

# Online: forward pass with TCS (N preserved)
for layer l in 1..L:
  X' = MHSA_pruned(LN(X)) + X;  Z = LN(X')
  sample S,  |S| ∈ [0.02N, 0.2N]
  I_FC1 = importance(Z, S)               # Eq.6 (DeiT) or Eq.7 (MAE/Swin)
  C1 = TopK(I_FC1, k1);  k1 = ⌊(1−r_FC1)D⌋
  Hh = GELU(Z[:,C1] @ W_FC1[C1,:])
  I_FC2 = importance(Hh, S)
  C2 = TopK(I_FC2, k2);  k2 = ⌊(1−r_FC2)·4D⌋
  X = Hh[:,C2] @ W_FC2[C2,:] + X'
return X
```

---

## 2. Per-layer pruning ratios (from paper Appendix E)

**DeiT-Small** (12 layers, MHSA 90% except L1):
| Layer | 1-7 | 8 | 9 | 10 | 11 | 12 |
|-------|-----|-----|-----|-----|-----|-----|
| FC1   | 0   | 0   | 0   | 0   | 0.5 | 0.5 |
| FC2   | 0   | 0.8 | 0.8 | 0.9 | 0.9 | 0.9 |

**ViT-MAE-Huge** (32 layers, MHSA 90% except L1; only FC2 in final 8):
| Layer | 1-24 | 25 | 26-31 | 32 |
|-------|------|-----|-------|-----|
| FC1   | 0    | 0   | 0     | 0.2 |
| FC2   | 0    | 0.8 | 0.9   | 0.9 |

**Swin-Base** (stages `[2,2,18,2]`, MHSA 90% except first layer of each stage):
FC1 = 0 everywhere except the last block (0.3). FC2 = 0.9 on the final ~11 blocks
of stage 2 + stage 3 (from block 14 onward), 0 before. See paper Table 13.

Full per-layer configs for all 9 models live in the official repo (this reimpl
should mirror them in `configs/`).

---

## 3. Requirements

### Environment
```
python >= 3.9
torch >= 2.0            # CUDA build
timm                    # DeiT / ViT-MAE / Swin model defs + pretrained weights
numpy, scipy            # geometric median, SVD/rank analysis
fvcore  (or ptflops)    # FLOPs counting
Pillow, pyyaml, tqdm
```
Downstream (optional, only for detection/segmentation tables):
```
mmdetection, mmsegmentation, mmcv     # Cascade Mask R-CNN (COCO), UPerNet (ADE20K)
pycocotools
```

### Hardware (to reproduce paper)
- Fine-tuning: 4× H100 (MHSA recovery). TCS needs **no** training / no GPU-hours.
- Benchmarking: single H100, batch 128, fp32.

### Datasets
- **ImageNet-1K** (classification, 224×224) — primary.
- COCO 2017 (detection, Cascade Mask R-CNN, short side 480–800 / long ≤1333).
- ADE20K (segmentation, 512×512).
- CIFAR-100 (224×224, transfer).

### Pretrained backbones (9 models, timm)
DeiT-{Tiny,Small,Base}; ViT-MAE-{Base,Large,Huge}; Swin-{Tiny,Small,Base}.

### Fine-tuning (MHSA recovery only)
- Optimizer AdamW, cosine LR schedule.
- Epochs are model-scale dependent (larger = fewer): DeiT-Small 290, ViT-MAE-Large
  139, ViT-MAE-Huge 15. Ablation fine-tunes used 50 epochs.

---

## 4. Proposed file structure

```
toast/
├── README.md
├── requirements.txt
├── configs/                      # per-model pruning ratios (all 9 models)
│   ├── deit_small.yaml           #   ρm, per-layer FC1/FC2 ratios, sampling range
│   ├── vit_mae_huge.yaml
│   ├── swin_base.yaml
│   └── ...
├── toast/
│   ├── __init__.py
│   ├── scwp.py                   # MHSA: coupled GM pruning (offline)
│   │                             #   build_coupled(W_Q,W_K,W_V,W_proj)
│   │                             #   geometric_median(W), coupled_importance()
│   │                             #   prune_head(dk→d'k), applies to a timm attn module
│   ├── tcs.py                    # FFN: token channel selection (online, training-free)
│   │                             #   sample_tokens(N, ratio)
│   │                             #   importance_cls(Eq6) / importance_mag(Eq7)
│   │                             #   select_channels(I,k); wraps FFN forward
│   ├── models.py                 # load timm backbones; inject SCWP + TCS hooks
│   ├── geometric_median.py       # Weiszfeld iteration (FPGM-style)
│   └── analysis.py               # sparsity, R² reconstruction, effective-rank, SNR
│                                 #   (reproduces Fig.4 / Tables 7,8 — optional)
├── scripts/
│   ├── prune_mhsa.py             # offline SCWP → save pruned checkpoint
│   ├── finetune.py               # AdamW+cosine recovery of pruned MHSA
│   ├── eval_imagenet.py          # apply TCS at inference, report Top-1/5, GFLOPs, throughput
│   └── benchmark.py              # H100 latency/throughput, FLOPs count
├── downstream/                   # optional: mmdet / mmseg configs + backbone wrappers
│   ├── coco_cascade_rcnn.py
│   └── ade20k_upernet.py
└── tests/
    └── test_scwp_tcs.py          # sanity: coupled index sync, GM shape, TopK reduction
```

### Module responsibilities (minimal)
- **scwp.py** — the only tricky correctness point is index synchronization
  (Q↔K share kept indices; V↔proj share kept indices). Get this wrong ⇒ collapse.
- **tcs.py** — pure inference wrapper. Recompute channel selection **per forward**
  from sampled tokens; slice weight rows/cols to dense sub-matrices. No grad.
- **analysis.py** — not needed for the method; only to reproduce the paper's
  redundancy figures/tables. Skip unless reproducing Fig.4 / SNR.

---

## 5. Key results to target (ImageNet-1K, H100 bs=128 fp32)

| Model | Top-1 base → ToaSt | GFLOPs (↓%) | Speedup |
|-------|--------------------|-------------|---------|
| DeiT-Small   | 79.82 → **83.40** | 4.6 → 2.5 (45.7%) | 2.07× |
| DeiT-Base    | 81.80 → **84.82** | 17.6 → 10.7 (39.2%) | 1.51× |
| ViT-MAE-Huge | 86.88 → **88.52** | 167.4 → 101.4 (39.4%) | 1.59× |
| Swin-Base    | 83.50 → **85.21** | 15.4 → 8.8 (42.7%) | 1.28× |

Ablation checks worth reproducing: MHSA-only *drops* accuracy (−2.6 to −4.2%p);
adding TCS *recovers past baseline* (+1.7 to +3.6%p) — TCS acts as a noise filter
(kept channels 3–5.5× higher SNR). Sampling: default ≈ full-token within 1.3%p.

---

## 6. Implementation order (suggested)

1. `geometric_median.py` + `scwp.py` with the coupled-index test — verify a pruned
   MHSA runs and shapes are consistent.
2. `finetune.py` on DeiT-Small (50 epochs) → reproduce MHSA-only ~76.1%.
3. `tcs.py` + `eval_imagenet.py` with DeiT-Small ratios → target 83.40%.
4. Add remaining 8 model configs.
5. (Optional) `analysis.py`, downstream tasks, ToMe composition.
```
```

Notes / open items the paper leaves to the codebase:
- Exact per-layer ratios for the other 6 models (mirror official repo).
- Sampling-rate-vs-depth mapping (paper gives only the [2%,20%] range).
- Fine-tuning hyperparameters beyond "AdamW + cosine" (LR, warmup, batch).
