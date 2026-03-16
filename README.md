# MiniLLM — Python Code Language Model From Scratch

A GPT-style language model trained on Python code, built entirely from scratch in PyTorch.
No HuggingFace Trainer. No pre-trained weights. Every component written by hand.

## Live Demo

Try the model: https://4721f2f906a49e70dd.gradio.live

## What's built from scratch

- **BPE Tokenizer** — Byte Pair Encoding with GPT-2 style regex pre-tokenization
- **Transformer** — Decoder-only, causal self-attention, Rotary Positional Embeddings (RoPE), SwiGLU FFN, pre-norm, weight tying
- **Training loop** — AdamW with weight decay, cosine LR schedule with warmup, mixed precision (fp16), gradient clipping, KV cache inference

## Model

| Parameter | Value |
|-----------|-------|
| Parameters | 15M |
| Layers | 6 |
| Attention heads | 6 |
| d_model | 384 |
| Context length | 256 tokens |
| Vocab size | 2048 |
| Activation | SwiGLU |
| Positional encoding | RoPE |

## Training

| Detail | Value |
|--------|-------|
| Dataset | CodeSearchNet Python (400k functions) |
| Epochs | 3 |
| Batch size | 32 |
| Learning rate | 3e-4 (cosine decay) |
| Val loss | 1.88 |
| Val perplexity | 6.59 |
| Hardware | Google Colab T4 GPU |

Perplexity of 6.59 means the model assigns high probability to real Python tokens —
it has genuinely learned Python syntax, indentation patterns, and common idioms.

## Sample outputs

> Note: This is a 15M parameter model trained from scratch on a single T4 GPU.
> It learns Python structure, syntax, and patterns — not perfect logic.
> Logic quality scales with model size and compute.

### Prompt: `def binary_search(arr, target):`
```python
def binary_search(arr, target):
    """
    Binary the sources of the given attributes as a geometry.
    Args:
        r: A list of tuples, [(binary, ..., ...)] object
        target: A list of the sources to search for each desired attachment.
    Returns:
        A list of tuples that can be returned.
    """
    def fixtore(fn):
        return [fn]
    return list(set(fn))
```

**What this shows:** The model has learned Python docstring format (Args, Returns),
nested function definitions, and indentation — all from scratch with no pre-training.
Logic quality scales with model size and compute.

## Architecture decisions

**Why RoPE over learned positional embeddings?**
Rotary embeddings encode relative position directly into the attention scores via rotation
matrices. They generalize better to sequence lengths not seen during training.

**Why SwiGLU over ReLU?**
SwiGLU uses a gating mechanism — `silu(W1*x) * W2*x` — that gives the network more
expressive control over information flow. Used in LLaMA and PaLM for this reason.

**Why weight tying?**
The input embedding matrix and the output LM head share weights. This cuts ~3M parameters,
acts as implicit regularization, and is theoretically motivated — the same token should
have similar representations at input and output.

**Why pre-norm over post-norm?**
`x = x + Attention(LayerNorm(x))` is more stable to train than applying LayerNorm after
the residual. Avoids gradient explosion at the start of training.

## Project structure
```
mini-llm/
├── MiniLLM_From_Scratch.ipynb   # full notebook — every cell copy-pasteable
├── tokenizer.json               # trained BPE tokenizer (2048 vocab)
└── README.md
```

## How to run

1. Open `MiniLLM_From_Scratch.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells in order — fully self-contained, no setup needed

## What I'd do with more compute

- Scale to 85M parameters (d_model=768, n_layers=12)
- Train on the full Stack dataset (30B tokens)
- Run a scaling laws experiment across 4 model sizes and compare to Chinchilla predictions
- Add LoRA fine-tuning on specific libraries (NumPy, PyTorch source)
- Implement speculative decoding for 2-3x faster inference
