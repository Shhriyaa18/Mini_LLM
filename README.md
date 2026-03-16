# MiniLLM — Python Code Language Model From Scratch

A GPT-style language model trained on Python code, built entirely from scratch in PyTorch.
No HuggingFace Trainer. No pre-trained weights. Every component written by hand.

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

### Prompt: `def binary_search(arr, target):`
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

### Prompt: `class Stack:`
```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0
```

## Architecture decisions

**Why RoPE over learned positional embeddings?**
Rotary embeddings encode relative position directly into the attention scores via rotation matrices.
They generalize better to sequence lengths not seen during training.

**Why SwiGLU over ReLU?**
SwiGLU uses a gating mechanism — `silu(W1*x) * W2*x` — that gives the network more expressive
control over information flow. Used in LLaMA and PaLM for this reason.

**Why weight tying?**
The input embedding matrix and the output LM head share weights. This cuts ~3M parameters,
acts as implicit regularization, and is theoretically motivated — the same token should have
similar representations at input and output.

**Why pre-norm over post-norm?**
`x = x + Attention(LayerNorm(x))` is more stable to train than applying LayerNorm after
the residual. Avoids gradient explosion at the start of training.

## Project structure
```
mini-llm/
├── MiniLLM_From_Scratch.ipynb   # full notebook
├── tokenizer.json               # trained BPE tokenizer
└── README.md
```

## How to run

Open the notebook in Google Colab with a T4 GPU runtime.
Run all cells in order — the notebook is fully self-contained.

## Live demo

Try the model: [Gradio demo link here]

## What I'd do with more compute

- Scale to 85M parameters (increase d_model to 768, n_layers to 12)
- Train on the full Stack dataset (30B tokens) instead of 400k functions
- Run a scaling laws experiment across 4 model sizes
- Add LoRA fine-tuning on specific libraries (NumPy, PyTorch)
- Implement speculative decoding for faster inference
