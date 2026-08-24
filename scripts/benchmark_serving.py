"""Benchmark the three schedulers against the same workload.

The workload matters as much as the code being measured. Two choices here are
deliberate:

Poisson arrivals. Dumping every request in at t=0 measures batch throughput on
a saturated queue, which no real system ever sees. Modelling arrivals at a
rate lets queueing delay appear, and queueing delay is where the schedulers
actually differ.

Skewed output lengths. If every request generates the same number of tokens,
static batching looks nearly as good as continuous batching -- nothing is
waiting on a straggler. Real traffic is heavily skewed, so this samples short
and long generations together. Uniform workloads flatter static batching and
hide the entire point.

    python scripts/benchmark_serving.py --ckpt runs/base/best.pt
    python scripts/benchmark_serving.py --random-init --rates 4 8 16
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from minillm.config import Config
from minillm.model import MiniLLM
from minillm.serving import SCHEDULERS, InferenceEngine, Request
from minillm.utils import load_checkpoint, pick_device, set_seed


def build_workload(n: int, rate: float, cfg, seed: int = 0) -> list[Request]:
    """n requests arriving as a Poisson process at `rate` requests/second."""
    rng = np.random.default_rng(seed)

    gaps = rng.exponential(1.0 / rate, size=n)
    arrivals = np.cumsum(gaps)

    max_prompt = min(64, cfg.context_len // 4)
    prompt_lens = rng.integers(8, max_prompt, size=n)

    # 70% short generations, 30% long -- roughly chat-shaped, and skewed
    # enough that head-of-line blocking has something to bite on.
    budget = cfg.context_len - max_prompt
    short = rng.integers(8, max(9, budget // 6), size=n)
    long = rng.integers(budget // 3, budget, size=n)
    output_lens = np.where(rng.random(n) < 0.7, short, long)

    return [
        Request(
            id=i,
            prompt_ids=rng.integers(0, cfg.vocab_size, size=int(prompt_lens[i])).tolist(),
            max_new_tokens=int(output_lens[i]),
            arrival_offset=float(arrivals[i]),
        )
        for i in range(n)
    ]


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


def summarize(name, rate, requests, stats, wall, slo) -> dict:
    ttfts = [r.ttft for r in requests if r.ttft is not None]
    e2es = [r.e2e_latency for r in requests if r.e2e_latency is not None]
    itls = [r.mean_itl for r in requests if r.mean_itl is not None]

    # Goodput: requests that finished inside the latency budget. Throughput
    # alone can be raised by letting tail latency collapse, so a scheduler
    # should be judged on how much of its output was actually served in time.
    within_slo = sum(1 for e in e2es if e <= slo)

    return {
        "scheduler": name,
        "arrival_rate": rate,
        "requests": len(requests),
        "wall_clock_s": round(wall, 3),
        "throughput_tok_s": round(stats.tokens_generated / wall, 1),
        "throughput_req_s": round(len(requests) / wall, 3),
        "goodput_req_s": round(within_slo / wall, 3),
        "mean_batch_size": round(stats.mean_batch_size, 2),
        "decode_steps": stats.decode_steps,
        "ttft_p50": round(pct(ttfts, 50), 4),
        "ttft_p99": round(pct(ttfts, 99), 4),
        "e2e_p50": round(pct(e2es, 50), 4),
        "e2e_p99": round(pct(e2es, 99), 4),
        "itl_mean": round(statistics.mean(itls), 5) if itls else float("nan"),
        "peak_fragmentation": round(stats.peak_fragmentation, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--rates", type=float, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--slo", type=float, default=5.0, help="e2e latency budget (s)")
    ap.add_argument("--out", default="benchmarks/serving.csv")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.device)
    set_seed(args.seed)

    if args.random_init:
        cfg = Config.from_yaml(args.config)
        model = MiniLLM(cfg.model).to(device).eval()
    else:
        model, cfg = load_checkpoint(args.ckpt, device)

    print(
        f"device={device}  requests={args.requests}  max_batch={args.batch_size}  "
        f"slo={args.slo}s\n"
    )

    rows = []
    header = (
        f"{'scheduler':<12}{'rate':>6}{'tok/s':>10}{'batch':>8}"
        f"{'ttft p50':>10}{'ttft p99':>10}{'e2e p99':>10}{'goodput':>9}"
    )

    for rate in args.rates:
        print(header)
        print("-" * len(header))
        for name in ("sequential", "static", "continuous"):
            requests = build_workload(args.requests, rate, cfg.model, seed=args.seed)
            scheduler = SCHEDULERS[name](max_batch_size=args.batch_size)
            engine = InferenceEngine(model, cfg.model, scheduler, device=device, temperature=0.0)

            import time

            t0 = time.perf_counter()
            engine.run(requests)
            wall = time.perf_counter() - t0

            row = summarize(name, rate, requests, engine.stats, wall, args.slo)
            rows.append(row)
            print(
                f"{name:<12}{rate:>6.0f}{row['throughput_tok_s']:>10.1f}"
                f"{row['mean_batch_size']:>8.2f}{row['ttft_p50']:>10.3f}"
                f"{row['ttft_p99']:>10.3f}{row['e2e_p99']:>10.3f}"
                f"{row['goodput_req_s']:>9.3f}"
            )
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")

    base = next(r for r in rows if r["scheduler"] == "sequential")
    best = max(rows, key=lambda r: r["throughput_tok_s"])
    print(
        f"\nBest: {best['scheduler']} at rate {best['arrival_rate']:.0f} -- "
        f"{best['throughput_tok_s']:.1f} tok/s vs {base['throughput_tok_s']:.1f} "
        f"sequential ({best['throughput_tok_s'] / base['throughput_tok_s']:.2f}x)"
    )


if __name__ == "__main__":
    main()
