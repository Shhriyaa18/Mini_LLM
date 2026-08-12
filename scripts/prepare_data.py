"""Train the tokenizer and pre-tokenize the corpus.

Run once. Tokenizing is slow and deterministic, so it should not sit inside
the training script -- caching it to disk means you can iterate on the model
without paying for it every time.

    python scripts/prepare_data.py --config configs/base.yaml --out data/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minillm.config import Config
from minillm.data import load_python_corpus, tokenize_corpus
from minillm.tokenizer import BPETokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.train.seed)

    print("Loading corpus...")
    texts = load_python_corpus(cfg.train)
    print(f"  {len(texts):,} functions, {sum(map(len, texts)):,} characters")

    random.shuffle(texts)
    n_val = int(cfg.train.val_fraction * len(texts))
    val_texts = texts[:n_val][: cfg.train.max_val_texts]
    train_texts = texts[n_val:][: cfg.train.max_train_texts]
    print(f"  train: {len(train_texts):,} | val: {len(val_texts):,}")

    print("\nTraining tokenizer...")
    tok = BPETokenizer(vocab_size=cfg.model.vocab_size)
    tok.train(train_texts[: cfg.train.tokenizer_train_texts])
    tok.save(out / "tokenizer.json")

    sample = "def fibonacci(n):\n    if n <= 1:\n        return n\n"
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample, "tokenizer round trip failed"
    print(
        f"  round trip OK | {len(sample)} chars -> {len(ids)} tokens "
        f"({len(ids) / len(sample):.2f} tok/char)"
    )

    print("\nTokenizing train split...")
    train_tokens = tokenize_corpus(train_texts, tok)
    print("Tokenizing val split...")
    val_tokens = tokenize_corpus(val_texts, tok)

    import numpy as np

    np.save(out / "train.npy", train_tokens)
    np.save(out / "val.npy", val_tokens)

    stats = {
        "train_texts": len(train_texts),
        "val_texts": len(val_texts),
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "train_bytes": sum(len(t.encode("utf-8")) for t in train_texts),
        "val_bytes": sum(len(t.encode("utf-8")) for t in val_texts),
        "vocab_size": len(tok),
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + json.dumps(stats, indent=2))
    print(f"\nWrote {out}/train.npy, {out}/val.npy, {out}/tokenizer.json")


if __name__ == "__main__":
    main()
