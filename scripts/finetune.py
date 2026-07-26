"""Recover the pruned-MHSA model with AdamW + cosine LR (paper Sec 4.1).

TCS needs no training; this only recovers SCWP. Epochs are model-scale
dependent (DeiT-S 290, MAE-L 139, MAE-H 15).
"""
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.data import create_dataset, create_loader, resolve_data_config

from toast.models import build_toast_model, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None, help="pruned checkpoint from prune_mhsa.py")
    ap.add_argument("--wandb-project", default=None, help="enable wandb logging under this project")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run = None
    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, job_type="finetune",
                         config={**vars(args), **cfg})
    # Fine-tune the SCWP-pruned model WITHOUT TCS (TCS is inference-only).
    model = build_toast_model(cfg, pretrained=True, scwp=True, tcs=False).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device)["state_dict"])

    dcfg = resolve_data_config({}, model=model)
    ds = create_dataset("", root=args.data_dir, split="train", is_training=True)
    loader = create_loader(ds, input_size=dcfg["input_size"], batch_size=args.batch_size,
                           is_training=True, no_aug=False, num_workers=args.workers,
                           **{k: dcfg[k] for k in ("mean", "std")})

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    step = 0
    for ep in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            # bf16 autocast: ~2.5x faster and ~2x less VRAM than fp32 on Ampere+.
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
            step += 1
            if run:
                run.log({"train/loss": loss.item(),
                         "train/lr": sched.get_last_lr()[0], "epoch": ep}, step=step)
        sched.step()
        avg = running / max(1, len(loader))
        print(f"epoch {ep + 1}/{args.epochs} loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e}")
        if run:
            run.log({"train/epoch_loss": avg, "epoch": ep + 1}, step=step)
    torch.save({"state_dict": model.state_dict(), "config": cfg}, args.out)
    print(f"saved finetuned checkpoint -> {args.out}")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
