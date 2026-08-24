"""The inference engine: model + cache + scheduler, driven by one loop.

Prefill is done one request at a time; only decode is batched. Real engines
batch prefill too (and chunk it, so a long prompt cannot stall decoding), but
separating the two keeps the scheduler comparison clean -- the only variable
across the three schedulers is which sequences decode together.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch

from ..cache import BatchedKVCache
from ..config import ModelConfig
from ..model import MiniLLM
from .request import Request, State
from .scheduler import Scheduler


@dataclass
class EngineStats:
    iterations: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    tokens_generated: int = 0
    # Summed batch occupancy over decode steps. Divided by decode_steps this
    # gives mean batch size, which is the single number that most directly
    # explains a throughput difference between schedulers.
    batch_occupancy: int = 0
    peak_fragmentation: float = 0.0

    @property
    def mean_batch_size(self) -> float:
        return self.batch_occupancy / max(1, self.decode_steps)


class InferenceEngine:
    def __init__(
        self,
        model: MiniLLM,
        cfg: ModelConfig,
        scheduler: Scheduler,
        device: str = "cpu",
        temperature: float = 0.0,
        top_k: Optional[int] = None,
    ):
        self.model = model.eval()
        self.cfg = cfg
        self.scheduler = scheduler
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.cache = BatchedKVCache(cfg, max_slots=scheduler.max_batch_size, device=device)
        self.stats = EngineStats()

    # ------------------------------------------------------------------ steps

    @torch.no_grad()
    def _prefill(self, req: Request, now: float) -> None:
        """Run the prompt through the model and seed the request's cache slot."""
        req.slot = self.cache.allocate()
        ids = torch.tensor([req.prompt_ids], dtype=torch.long, device=self.device)
        positions = torch.arange(req.prompt_len, device=self.device).unsqueeze(0)
        slots = torch.tensor([req.slot], dtype=torch.long, device=self.device)

        logits, _ = self.model(ids, positions=positions, cache=self.cache, slots=slots)
        token = self._sample(logits[:, -1, :])

        req.output_ids.append(int(token))
        req.state = State.RUNNING
        req.first_token_at = now
        req.token_times.append(now)
        self.stats.prefill_steps += 1
        self.stats.tokens_generated += 1

    @torch.no_grad()
    def _decode_batch(self, batch: list[Request], now: float) -> None:
        """One decode step for every running request, as a single forward pass."""
        ids = torch.tensor(
            [[r.output_ids[-1]] for r in batch], dtype=torch.long, device=self.device
        )
        positions = torch.tensor(
            [[r.position] for r in batch], dtype=torch.long, device=self.device
        )
        slots = torch.tensor([r.slot for r in batch], dtype=torch.long, device=self.device)

        logits, _ = self.model(ids, positions=positions, cache=self.cache, slots=slots)
        tokens = self._sample(logits[:, -1, :])

        for req, tok in zip(batch, tokens.tolist()):
            req.output_ids.append(int(tok))
            req.token_times.append(now)

        self.stats.decode_steps += 1
        self.stats.batch_occupancy += len(batch)
        self.stats.tokens_generated += len(batch)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0:
            return logits.argmax(dim=-1)
        logits = logits / self.temperature
        if self.top_k:
            k = min(self.top_k, logits.size(-1))
            threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # ------------------------------------------------------------------- loop

    def run(self, requests: list[Request]) -> list[Request]:
        """Serve every request, honouring arrival times. Blocks until all done."""
        pending = sorted(requests, key=lambda r: r.arrival_offset)
        start = time.perf_counter()
        i = 0

        while i < len(pending) or self.scheduler.has_work:
            now = time.perf_counter() - start

            # Admit anything that has arrived by now.
            while i < len(pending) and pending[i].arrival_offset <= now:
                pending[i].arrived_at = now
                self.scheduler.add(pending[i])
                i += 1

            admit, evict = self.scheduler.schedule()

            for req in evict:
                req.state = State.DONE
                req.finished_at = time.perf_counter() - start
                if req.slot is not None:
                    self.cache.free(req.slot)
                    req.slot = None

            for req in admit:
                self._prefill(req, time.perf_counter() - start)
                self.scheduler.running.append(req)

            active = [r for r in self.scheduler.running if not r.is_finished]
            if active:
                self._decode_batch(active, time.perf_counter() - start)
                self.stats.peak_fragmentation = max(
                    self.stats.peak_fragmentation, self.cache.fragmentation()
                )
            elif not self.scheduler.running and i < len(pending):
                # Idle: nothing running and the next arrival is in the future.
                time.sleep(0.001)

            self.stats.iterations += 1

        return requests
