"""Corpus loading and the packed token dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import TrainConfig
from .tokenizer import BPETokenizer


def load_python_corpus(cfg: TrainConfig, split: str = "train") -> list[str]:
    """Pull Python functions out of CodeSearchNet."""
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=split)
    texts: list[str] = []
    for ex in ds:
        code = ex.get("whole_func_string") or ex.get("func_code_string") or ""
        if len(code) >= cfg.min_text_chars:
            texts.append(code)
    return texts


def tokenize_corpus(
    texts: list[str],
    tokenizer: BPETokenizer,
    separator_id: int = 0,
    log_every: int = 5_000,
) -> np.ndarray:
    """Tokenize into one flat array, separating documents with a single id."""
    chunks: list[list[int]] = []
    for i, text in enumerate(texts, 1):
        ids = tokenizer.encode(text)
        ids.append(separator_id)
        chunks.append(ids)
        if log_every and i % log_every == 0:
            print(f"  tokenized {i:,}/{len(texts):,}")

    flat = np.concatenate([np.asarray(c, dtype=np.uint16) for c in chunks])
    return flat


class PackedDataset(Dataset):
    """Fixed-length windows over a flat token stream.

    Packing rather than padding means no wasted compute: every position in
    every batch is a real training target.
    """

    def __init__(self, tokens: np.ndarray, context_len: int):
        if len(tokens) <= context_len:
            raise ValueError(
                f"corpus has {len(tokens)} tokens, need more than context_len={context_len}"
            )
        self.tokens = torch.from_numpy(tokens.astype(np.int64))
        self.context_len = context_len

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.context_len

    def __getitem__(self, idx: int):
        start = idx * self.context_len
        x = self.tokens[start : start + self.context_len]
        y = self.tokens[start + 1 : start + self.context_len + 1]
        return x, y

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.tokens.numpy().astype(np.uint16))

    @classmethod
    def load(cls, path: str | Path, context_len: int) -> "PackedDataset":
        return cls(np.load(path), context_len)
