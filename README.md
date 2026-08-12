# MiniLLM

A 15M-parameter decoder-only transformer trained on Python source, built from
scratch in PyTorch. No Hugging Face, no pretrained weights, no reference
implementation — tokenizer, attention, positional embeddings, training loop,
and KV-cached inference are all written here.

Built to understand transformer internals end to end. The results below are
what a model this size can do; the point is the implementation, not the model.

---

## Results

| Metric | Value |
|---|---|
| Parameters | 15.0M total / 14.2M non-embedding |
| Validation perplexity | _fill in from `runs/base/summary.json`_ |
| Validation bits/byte | _fill in from `scripts/evaluate.py`_ |
| KV-cache speedup @ 192 tokens | _fill in from `benchmarks/kv_cache.csv`_ |
| Training corpus | CodeSearchNet Python |

Perplexity is per token and therefore tokenizer-dependent — a smaller vocabulary
splits text into more, individually easier tokens and reports a lower number
without the model being better. Bits-per-byte normalizes that out and is the
figure to compare across tokenizers.

## Architecture

LLaMA-flavoured decoder-only transformer.

| Component | Choice |
|---|---|
| Tokenizer | Byte-level BPE, 2048 merges, GPT-2 regex pre-tokenizer |
| Positions | Rotary embeddings (RoPE) applied to Q and K |
| Attention | Multi-head causal self-attention, KV cache for decoding |
| Feed-forward | SwiGLU, `d_ff = 4 × d_model` |
| Norm | Pre-norm LayerNorm |
| Embeddings | Input/output weights tied |
| Layers / heads / `d_model` | 6 / 6 / 384 |
| Context | 256 tokens |

Training uses AdamW with decoupled weight decay applied to matrices only,
cosine decay with linear warmup, gradient clipping at 1.0, and mixed precision
on CUDA.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_data.py --config configs/base.yaml --out data
python scripts/train.py --config configs/base.yaml --data data --out runs/base
python scripts/evaluate.py --ckpt runs/base/best.pt --data data
python scripts/generate.py --ckpt runs/base/best.pt --prompt "def binary_search(arr, target):"
python scripts/benchmark_generation.py --ckpt runs/base/best.pt
```

Tests run without a checkpoint or dataset:

```bash
pytest tests/ -q
```

## Layout

```
src/minillm/
  config.py      dataclass configs, YAML round trip, stored in checkpoints
  tokenizer.py   byte-level BPE: training, rank-ordered encoding, serialization
  model.py       RoPE, causal attention with KV cache, SwiGLU, MiniLLM
  data.py        corpus loading, tokenization, packed fixed-length windows
  trainer.py     training loop, LR schedule, AMP, checkpointing
  utils.py       seeding, device selection, checkpoint loading
scripts/         prepare_data, train, evaluate, generate, benchmark_generation
tests/           tokenizer and model correctness tests
configs/base.yaml
```

## Implementation notes

Three things here were harder than they look, and each has a test guarding it.

**The KV cache has to agree with a full forward pass.** Caching keys and values
turns decoding from O(n²) total work into O(n), but two bugs make it wrong
without making it crash. RoPE must rotate each token at its *absolute*
position — during cached decoding the input tensor has length 1, so rotating
"from position 0" would stack every generated token at the start of the
sequence. And the causal mask must be indexed by absolute query position
against absolute key position; slicing the top-left corner of a triangular
mask is only correct when the cache is empty. Both produce fluent-looking
degraded output rather than an error.
`tests/test_model.py::test_kv_cache_matches_full_forward` asserts cached
decoding reproduces the uncached logits to 1e-5.

**BPE encoding has to apply merges in rank order.** The obvious implementation
scans left to right and applies the first merge it finds. That can segment
text differently than training did, which silently costs compression and
introduces a train/inference mismatch. Encoding here repeatedly selects the
pair with the lowest merge index instead.

**The pre-tokenizer regex has to cover every character.** The branches must
partition the full character set, or unmatched characters are dropped with no
error. Using `[^\s\w]` as the catch-all class looks right but excludes
underscore and every non-ASCII letter, because `\w` contains them — so
`__init__` pre-tokenizes to `init`. On Python source that removes a large
fraction of every identifier. The correct catch-all is `[^\s\p{L}\p{N}]`, and
`tests/test_tokenizer.py::test_pretokenizer_is_lossless` asserts the
concatenated pre-tokens reconstruct the input exactly.

## Roadmap

Continuous batching and a paged KV cache, benchmarked as a
throughput-versus-p99-latency curve.
