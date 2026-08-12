"""Sample from a checkpoint.

    python scripts/generate.py --ckpt runs/base/best.pt --prompt "def binary_search(arr, target):"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from minillm.tokenizer import BPETokenizer
from minillm.utils import load_checkpoint, pick_device, set_seed

DEFAULT_PROMPTS = [
    "def binary_search(arr, target):",
    "class Stack:\n    def __init__(self):",
    'def merge_sort(arr):\n    """Sort an array using merge sort."""',
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    device = pick_device(args.device)
    model, _ = load_checkpoint(args.ckpt, device)
    tok = BPETokenizer.load(args.tokenizer)

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    for prompt in prompts:
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(
            ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print("=" * 70)
        print(tok.decode(out[0].tolist()))
        print()


if __name__ == "__main__":
    main()
