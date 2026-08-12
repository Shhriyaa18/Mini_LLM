"""Configuration objects for MiniLLM.

Everything tunable lives here or in a YAML file, never inline in a training
script. This is what makes runs reproducible: a checkpoint stores the exact
config it was trained with, so loading a model never depends on remembering
which constants were set at the time.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""

    vocab_size: int = 2048
    context_len: int = 256
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    d_ff: int = 1536
    dropout: float = 0.1
    bias: bool = False
    # Use PyTorch's fused scaled_dot_product_attention instead of the manual
    # implementation. Mathematically identical (tests/test_model.py asserts
    # this), but much faster. The manual path is kept because it is the one
    # worth reading and explaining.
    use_sdpa: bool = True

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.d_head % 2 != 0:
            raise ValueError(f"d_head ({self.d_head}) must be even for RoPE")


@dataclass
class TrainConfig:
    """Optimization and data hyperparameters."""

    # data
    dataset_name: str = "code_search_net"
    dataset_config: str = "python"
    max_train_texts: int = 30_000
    max_val_texts: int = 2_000
    val_fraction: float = 0.05
    min_text_chars: int = 50

    # tokenizer
    tokenizer_train_texts: int = 3_000

    # optimization
    batch_size: int = 32
    max_epochs: int = 3
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_fraction: float = 0.05
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)

    # loop
    eval_every: int = 200
    eval_batches: int = 50
    log_every: int = 50
    num_workers: int = 2
    seed: int = 1337
    amp: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        return {"model": asdict(self.model), "train": asdict(self.train)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        model_keys = {f.name for f in fields(ModelConfig)}
        train_keys = {f.name for f in fields(TrainConfig)}
        return cls(
            model=ModelConfig(**{k: v for k, v in raw.get("model", {}).items() if k in model_keys}),
            train=TrainConfig(**{k: v for k, v in raw.get("train", {}).items() if k in train_keys}),
        )
