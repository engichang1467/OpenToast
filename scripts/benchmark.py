"""Throughput / latency benchmark (paper: single H100, bs=128, fp32)."""
import argparse
import time
import torch

from toast.models import build_toast_model, load_config


@torch.no_grad()
def throughput(model, input_size, batch_size, device, warmup=10, iters=50):
    model.eval()
    x = torch.zeros(batch_size, *input_size, device=device)
    for _ in range(warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return batch_size * iters / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--baseline", action="store_true", help="no compression")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compress = not args.baseline
    model = build_toast_model(cfg, pretrained=False, scwp=compress, tcs=compress).to(device)
    from timm.data import resolve_data_config
    isize = resolve_data_config({}, model=model)["input_size"]
    ips = throughput(model, isize, args.batch_size, device)
    print(f"{'baseline' if args.baseline else 'ToaSt'}: {ips:.1f} img/s")


if __name__ == "__main__":
    main()
