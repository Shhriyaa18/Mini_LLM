"""Training loop."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config
from .model import MiniLLM


class Trainer:
    def __init__(
        self,
        model: MiniLLM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Config,
        out_dir: str | Path,
        device: str = "cpu",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.tcfg = cfg.train
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.best_val = float("inf")
        self.total_steps = self.tcfg.max_epochs * len(train_loader)
        self.warmup_steps = max(1, int(self.tcfg.warmup_fraction * self.total_steps))
        self.history: list[dict] = []

        # Weight decay on matrices only. Biases, LayerNorm gains, and anything
        # else 1-D is excluded -- decaying those hurts and is a common bug.
        decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.tcfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.tcfg.lr,
            betas=tuple(self.tcfg.betas),
        )

        self.use_amp = self.tcfg.amp and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # ------------------------------------------------------------------ utils

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.tcfg.lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        return self.tcfg.min_lr + 0.5 * (self.tcfg.lr - self.tcfg.min_lr) * (
            1 + math.cos(math.pi * progress)
        )

    def _autocast(self):
        return torch.amp.autocast("cuda", enabled=self.use_amp)

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> float:
        self.model.eval()
        limit = max_batches or self.tcfg.eval_batches
        total, n = 0.0, 0
        for i, (x, y) in enumerate(self.val_loader):
            if i >= limit:
                break
            x, y = x.to(self.device), y.to(self.device)
            with self._autocast():
                _, loss = self.model(x, y)
            total += loss.item()
            n += 1
        self.model.train()
        return total / max(1, n)

    def save_checkpoint(self, name: str, val_loss: float) -> Path:
        path = self.out_dir / name
        torch.save(
            {
                "model": self.model.state_dict(),
                "config": self.cfg.to_dict(),
                "step": self.step,
                "val_loss": val_loss,
                "val_ppl": math.exp(val_loss),
            },
            path,
        )
        return path

    # ------------------------------------------------------------------- loop

    def train(self) -> dict:
        self.model.train()
        t0 = time.time()
        running = 0.0
        running_n = 0

        print(
            f"Training {self.model.num_params() / 1e6:.2f}M params for "
            f"{self.total_steps:,} steps on {self.device}"
        )

        for epoch in range(self.tcfg.max_epochs):
            for x, y in self.train_loader:
                lr = self.lr_at(self.step)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                with self._autocast():
                    _, loss = self.model(x, y)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running += loss.item()
                running_n += 1
                self.step += 1

                if self.step % self.tcfg.log_every == 0:
                    avg = running / running_n
                    running, running_n = 0.0, 0
                    elapsed = time.time() - t0
                    print(
                        f"epoch {epoch} | step {self.step:,}/{self.total_steps:,} "
                        f"| loss {avg:.4f} | ppl {math.exp(avg):.2f} "
                        f"| lr {lr:.2e} | {elapsed:.0f}s"
                    )
                    self.history.append(
                        {"step": self.step, "split": "train", "loss": avg, "lr": lr}
                    )

                if self.step % self.tcfg.eval_every == 0:
                    val_loss = self.evaluate()
                    tag = ""
                    if val_loss < self.best_val:
                        self.best_val = val_loss
                        self.save_checkpoint("best.pt", val_loss)
                        tag = "  <- best"
                    print(
                        f"  eval  | step {self.step:,} | val loss {val_loss:.4f} "
                        f"| val ppl {math.exp(val_loss):.2f}{tag}"
                    )
                    self.history.append(
                        {"step": self.step, "split": "val", "loss": val_loss}
                    )

        final_val = self.evaluate(max_batches=None)
        self.save_checkpoint("final.pt", final_val)

        with open(self.out_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        summary = {
            "steps": self.step,
            "best_val_loss": round(self.best_val, 4),
            "best_val_ppl": round(math.exp(self.best_val), 2),
            "final_val_loss": round(final_val, 4),
            "final_val_ppl": round(math.exp(final_val), 2),
            "wall_clock_s": round(time.time() - t0, 1),
        }
        with open(self.out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + json.dumps(summary, indent=2))
        return summary
