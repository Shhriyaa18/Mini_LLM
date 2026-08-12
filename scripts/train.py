"""Train MiniLLM.

    python scripts/train.py --config configs/base.yaml --data data --out runs/base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from minillm.config import Config
from minillm.data import PackedDataset
from minillm.model import MiniLLM
from minillm.trainer import Trainer
from minillm.utils import pick_device, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/base")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.train.seed)
    device = pick_device(args.device)

    data = Path(args.data)
    train_ds = PackedDataset.load(data / "train.npy", cfg.model.context_len)
    val_ds = PackedDataset.load(data / "val.npy", cfg.model.context_len)
    print(
        f"train: {train_ds.num_tokens:,} tokens / {len(train_ds):,} windows | "
        f"val: {val_ds.num_tokens:,} tokens / {len(val_ds):,} windows"
    )

    pin = device == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=pin,
    )

    model = MiniLLM(cfg.model).to(device)
    print(
        f"params: {model.num_params()/1e6:.2f}M total, "
        f"{model.num_params(non_embedding=True)/1e6:.2f}M non-embedding"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "config.yaml")

    Trainer(model, train_loader, val_loader, cfg, out_dir, device).train()


if __name__ == "__main__":
    main()
