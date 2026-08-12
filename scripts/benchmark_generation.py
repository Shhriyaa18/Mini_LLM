"""Measure what the KV cache actually buys.

Without a cache, every decode step re-runs attention over the entire prefix,
so generating n tokens costs O(n^2) work. With a cache, each step attends the
new token against stored keys and values: O(n). This script measures the gap
and writes a CSV you can plot.

    python scripts/benchmark_generation.py --ckpt runs/base/best.pt

Works without a checkpoint too (random weights) -- latency does not depend on
whether the model is trained:

    python scripts/benchmark_generation.py --random-init
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from minillm.config import Config
from minillm.model import MiniLLM
from minillm.utils import load_checkpoint, pick_device, set_seed


def time_generation(model, prompt_ids, n_tokens, use_cache, device, repeats=3):
    latencies = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(
            prompt_ids,
            max_new_tokens=n_tokens,
            temperature=0.8,
            top_k=50,
            use_cache=use_cache,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
    return statistics.median(latencies)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="benchmarks/kv_cache.csv")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    set_seed(0)

    if args.random_init:
        cfg = Config.from_yaml(args.config)
        model = MiniLLM(cfg.model).to(device).eval()
    else:
        model, cfg = load_checkpoint(args.ckpt, device)

    max_new = cfg.model.context_len - args.prompt_len
    lengths = [n for n in (16, 32, 64, 128, 192) if n <= max_new]

    prompt_ids = torch.randint(
        0, cfg.model.vocab_size, (1, args.prompt_len), device=device
    )

    # Warm up: first call pays for lazy CUDA init and kernel autotuning.
    model.generate(prompt_ids, max_new_tokens=8, use_cache=True)
    model.generate(prompt_ids, max_new_tokens=8, use_cache=False)

    rows = []
    print(f"device={device}  prompt_len={args.prompt_len}  repeats={args.repeats}\n")
    print(f"{'tokens':>8} {'no cache (s)':>14} {'cache (s)':>12} {'speedup':>9} {'tok/s':>9}")
    print("-" * 56)

    for n in lengths:
        naive = time_generation(model, prompt_ids, n, False, device, args.repeats)
        cached = time_generation(model, prompt_ids, n, True, device, args.repeats)
        speedup = naive / cached
        rows.append(
            {
                "new_tokens": n,
                "no_cache_s": round(naive, 4),
                "cache_s": round(cached, 4),
                "speedup": round(speedup, 2),
                "cached_tok_per_s": round(n / cached, 1),
            }
        )
        print(
            f"{n:>8} {naive:>14.4f} {cached:>12.4f} {speedup:>8.2f}x "
            f"{n / cached:>8.1f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {out}")
    print(
        "Note: the gap widens with sequence length -- that is the O(n^2) vs O(n) "
        "difference showing up, not a constant-factor win."
    )


if __name__ == "__main__":
    main()
