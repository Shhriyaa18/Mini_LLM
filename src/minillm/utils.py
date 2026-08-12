"""Shared helpers: seeding, device selection, checkpoint loading."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .model import MiniLLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[MiniLLM, Config]:
    """Rebuild a model from a checkpoint without needing the training script."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = MiniLLM(cfg.model)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg
