# Notebook -> repo: what changed and what to re-run

## Bugs found in the notebook

### 1. Pre-tokenizer dropped underscores and all non-ASCII characters (critical)

The pattern used `[^\s\w]+` as its catch-all class. `\w` includes underscore and
Unicode letters, so those characters matched no branch and were silently
discarded. On Python source this is severe: `__init__` became `init`,
`max_len` became `maxlen`, `self._cache` became `self.cache`.

The model never saw an underscore. Every identifier in the training corpus was
corrupted, and reported perplexity was measured on that corrupted corpus.

**Consequence: the tokenizer and the model both have to be retrained, and the
perplexity number on the resume has to be re-measured.**

Fixed by using the real GPT-2 class set: `[^\s\p{L}\p{N}]+`.
Guarded by `test_pretokenizer_is_lossless`.

### 2. The KV cache was never exercised, and was wrong

`generate()` called `self(idx_cond)` without `use_cache=True`, so the cache
code never ran — generation recomputed the entire prefix every step. Had it
been switched on, two bugs would have produced silently degraded output:

- RoPE was applied as `self.rope(q, T)` with `T` the tensor length. During
  cached decoding `T == 1`, so every generated token was rotated to position 0.
- The mask slice `causal_mask[:, :, :T, :T_k]` with `T == 1` selects the first
  row of the triangular mask, `[1, 0, 0, ...]`, which lets the new token attend
  only to position 0.

Fixed by threading an explicit `pos_offset` through the forward pass.
Guarded by `test_kv_cache_matches_full_forward`.

### 3. BPE encoding merged greedily instead of by rank

Encoding applied whichever merge it found first scanning left to right, rather
than the lowest-rank merge. This can segment text differently than training
did. Fixed to select by merge index.

### 4. Deprecated AMP API

`torch.cuda.amp.GradScaler` / `autocast` are deprecated in favour of
`torch.amp.GradScaler("cuda")` / `torch.amp.autocast("cuda")`.

## Structural changes

- Two conflicting `Trainer` definitions (cells 19 and 29) collapsed into one.
- Hardcoded `/content/` and Google Drive paths replaced with CLI arguments.
- Config extracted to `configs/base.yaml`; the config is saved inside every
  checkpoint, so loading a model never depends on remembering hyperparameters.
- Tokenization moved into `scripts/prepare_data.py` and cached as `.npy`, so
  training runs do not re-pay for it.
- Seeding added across `random`, `numpy`, and `torch`.
- Scaled residual initialization (GPT-2 recipe) added.
- Bits-per-byte added alongside perplexity in `scripts/evaluate.py`.
- `benchmarks/` and a benchmark script added, measuring cached versus uncached
  decoding.
- 21 tests added.

## Tokenizer training speedup

Merge statistics are now computed over unique pre-tokens with a count, rather
than over every occurrence. Python source is extremely repetitive (`    `,
`self`, `def`, `return`), so this collapses the corpus substantially and makes
it practical to train the tokenizer on far more than 3,000 files.

## Re-run order

```bash
pytest tests/ -q
python scripts/prepare_data.py --config configs/base.yaml --out data
python scripts/train.py --config configs/base.yaml --data data --out runs/base
python scripts/evaluate.py --ckpt runs/base/best.pt --data data
python scripts/benchmark_generation.py --ckpt runs/base/best.pt
```

Then fill the results table in `README.md` from `runs/base/summary.json`,
the evaluate output, and `benchmarks/kv_cache.csv`.

## Resume accuracy

The notebook set `max_texts=30000` on the training dataset and trained the
tokenizer on 3,000 texts. The corpus was filtered from CodeSearchNet's ~400k
Python functions, but 30,000 is what the model actually trained on. The resume
line "trained on 400,000 Python functions" does not match the code, and the
config is visible in the repo.

Either raise `max_train_texts` and retrain, or change the line to match what
you ran.
