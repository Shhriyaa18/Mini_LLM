"""Byte-level BPE tokenizer, built from scratch.

Two things here differ from the naive textbook implementation, and both are
worth being able to explain:

1. Training buckets identical pre-tokens and counts them once with a weight,
   instead of walking every occurrence in the corpus on every merge. Code has
   an extremely skewed token distribution ("    ", "self", "def", "return"),
   so this collapses the corpus by roughly an order of magnitude and makes the
   merge loop tractable on more than a few thousand files.

2. Encoding applies merges in *rank order* (lowest merge index first), not
   greedily left-to-right. The naive version produces a different segmentation
   than training did for some inputs, which silently costs you compression and
   introduces a train/inference mismatch.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import regex

# GPT-2 style pre-tokenizer split. Keeps merges from ever crossing a word or
# whitespace boundary, which is what stops the vocabulary filling up with
# junk like "):\n    return".
#
# The three content classes must together cover every non-whitespace character,
# or the pre-tokenizer silently drops input. `[^\s\p{L}\p{N}]` is the catch-all
# that picks up punctuation *and* underscore -- using `[^\s\w]` there instead
# looks equivalent but is not, because `\w` includes underscore and every
# Unicode letter, so those characters match no branch at all and vanish.
# For Python source that is severe: `__init__` pre-tokenizes to `init`.
GPT2_PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    def __init__(self, vocab_size: int = 2048):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (one id per byte)")
        self.vocab_size = vocab_size
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._encode_cache: dict[str, list[int]] = {}

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _pre_tokenize(text: str) -> list[str]:
        return regex.findall(GPT2_PATTERN, text)

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out: list[int] = []
        i = 0
        a, b = pair
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # --------------------------------------------------------------- training

    def train(self, texts: Iterable[str], verbose: bool = True) -> None:
        # Bucket identical pre-tokens and carry a count, rather than storing
        # one list per occurrence.
        word_counts: dict[str, int] = defaultdict(int)
        n_texts = 0
        for text in texts:
            n_texts += 1
            for chunk in self._pre_tokenize(text):
                word_counts[chunk] += 1

        words: list[list[int]] = [list(w.encode("utf-8")) for w in word_counts]
        counts: list[int] = list(word_counts.values())
        total_tokens = sum(len(w) * c for w, c in zip(words, counts))

        if verbose:
            print(
                f"Corpus: {n_texts:,} texts | {len(words):,} unique pre-tokens "
                f"| {total_tokens:,} bytes"
            )

        num_merges = self.vocab_size - 256
        for step in range(num_merges):
            stats: dict[tuple[int, int], int] = defaultdict(int)
            for word, count in zip(words, counts):
                for pair in zip(word, word[1:]):
                    stats[pair] += count

            if not stats:
                if verbose:
                    print(f"No pairs left after {step} merges; stopping early.")
                break

            best = max(stats, key=stats.get)
            new_id = 256 + step
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            words = [self._merge(w, best, new_id) for w in words]

            if verbose and (step + 1) % 200 == 0:
                preview = self.vocab[new_id].decode("utf-8", errors="replace")
                print(
                    f"  merge {step + 1}/{num_merges}: {best} -> {new_id} "
                    f"({preview!r}) freq={stats[best]:,}"
                )

        self._encode_cache.clear()
        if verbose:
            print(f"Done. Vocabulary size: {len(self.vocab):,}")

    # --------------------------------------------------------------- encoding

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._encode_cache.get(chunk)
        if cached is not None:
            return cached

        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Lowest merge rank wins, matching the order merges were learned.
            pairs = set(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = self._merge(ids, best, self.merges[best])

        self._encode_cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in self._pre_tokenize(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        raw = b"".join(self.vocab.get(int(i), b"") for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ---------------------------------------------------------- serialization

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            # Merges are rank-ordered; the list preserves that order explicitly
            # rather than relying on dict insertion order surviving a round trip.
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        with open(path) as f:
            payload = json.load(f)
        tok = cls(vocab_size=payload["vocab_size"])
        for a, b, idx in payload["merges"]:
            tok.merges[(a, b)] = idx
            tok.vocab[idx] = tok.vocab[a] + tok.vocab[b]
        return tok

    def __len__(self) -> int:
        return len(self.vocab)
