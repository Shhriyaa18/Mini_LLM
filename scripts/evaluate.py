"""Evaluate a checkpoint: loss, perplexity, and bits-per-byte.

Perplexity is per *token*, so it depends on the tokenizer -- a smaller vocab
splits text into more, easier-to-predict tokens and reports a lower number
without the model being better. Bits-per-byte normalizes that away and is the
figure to quote when comparing against anything trained with a different
tokenizer.

    python scripts/evaluate.py --ckpt runs/base/best.pt --data data
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from minillm.data import PackedDataset
from minillm.utils import load_checkpoint, pick_device


@torch.no_grad()
def eval_loss(model, loader, device, max_batches=None):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total += loss.item()
        n += 1
    return total / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = pick_device(args.device)
    model, cfg = load_checkpoint(args.ckpt, device)

    data = Path(args.data)
    val_ds = PackedDataset.load(data / "val.npy", cfg.model.context_len)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    loss = eval_loss(model, loader, device)
    ppl = math.exp(loss)

    result = {"val_loss": round(loss, 4), "val_ppl": round(ppl, 2)}

    stats_path = data / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        # bits/byte = (nats/token) * (tokens/byte) / ln(2)
        tokens_per_byte = stats["val_tokens"] / stats["val_bytes"]
        result["bits_per_byte"] = round(loss * tokens_per_byte / math.log(2), 4)
        result["tokens_per_byte"] = round(tokens_per_byte, 4)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
