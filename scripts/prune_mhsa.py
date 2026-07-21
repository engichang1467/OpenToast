"""Offline SCWP: prune MHSA and save the pruned (pre-finetune) checkpoint."""
import argparse
import torch

from toast.models import build_toast_model, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-classes", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    # SCWP only; TCS is applied later at inference time (training-free).
    model = build_toast_model(cfg, pretrained=True,
                              num_classes=args.num_classes, scwp=True, tcs=False)
    torch.save({"state_dict": model.state_dict(), "config": cfg}, args.out)
    print(f"saved pruned MHSA checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
