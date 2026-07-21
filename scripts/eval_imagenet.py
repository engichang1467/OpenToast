"""Evaluate a ToaSt model on ImageNet-1K val: Top-1/5 + GFLOPs.

Applies SCWP (or loads a finetuned checkpoint) + TCS, then runs the val loop.
"""
import argparse
import torch
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.utils import accuracy, AverageMeter

from toast.models import build_toast_model, load_config
from toast.tcs import apply_tcs


def gflops(model, input_size):
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    model.eval()
    x = torch.zeros(1, *input_size, device=next(model.parameters()).device)
    return FlopCountAnalysis(model, x).total() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--checkpoint", default=None, help="finetuned SCWP checkpoint")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.checkpoint:
        # Load finetuned pruned weights, then attach TCS on top.
        model = build_toast_model(cfg, pretrained=True, scwp=True, tcs=False)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])
        apply_tcs(model, cfg["fc1_ratios"], cfg["fc2_ratios"],
                  use_attn=cfg.get("use_attn", False),
                  sample_lo=cfg.get("sample_lo", 0.02), sample_hi=cfg.get("sample_hi", 0.20))
    else:
        model = build_toast_model(cfg, pretrained=True, scwp=True, tcs=True)
    model = model.to(device).eval()

    dcfg = resolve_data_config({}, model=model)
    ds = create_dataset("", root=args.data_dir, split="val", is_training=False)
    loader = create_loader(ds, input_size=dcfg["input_size"], batch_size=args.batch_size,
                           is_training=False, crop_pct=dcfg["crop_pct"],
                           **{k: dcfg[k] for k in ("mean", "std")})

    top1, top5 = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            a1, a5 = accuracy(model(x), y, topk=(1, 5))
            top1.update(a1.item(), x.size(0)); top5.update(a5.item(), x.size(0))

    g = gflops(model, dcfg["input_size"])
    print(f"Top-1 {top1.avg:.2f}  Top-5 {top5.avg:.2f}  GFLOPs {g if g else 'n/a'}")


if __name__ == "__main__":
    main()
