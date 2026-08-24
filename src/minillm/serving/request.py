"""Request state and the metrics a serving system is judged on."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class State(enum.Enum):
    WAITING = "waiting"    # arrived, not yet admitted to the batch
    RUNNING = "running"    # prefilled, occupying a cache slot, decoding
    DONE = "done"


@dataclass
class Request:
    id: int
    prompt_ids: list[int]
    max_new_tokens: int
    # Seconds after benchmark start when this request arrives. Modelling
    # arrivals rather than dumping every request in at t=0 is what makes
    # queueing delay -- and therefore the whole scheduling problem -- visible.
    arrival_offset: float = 0.0

    state: State = State.WAITING
    slot: Optional[int] = None
    output_ids: list[int] = field(default_factory=list)

    arrived_at: Optional[float] = None
    first_token_at: Optional[float] = None
    finished_at: Optional[float] = None
    token_times: list[float] = field(default_factory=list)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def num_generated(self) -> int:
        return len(self.output_ids)

    @property
    def position(self) -> int:
        """Absolute position of the next token to be generated."""
        return self.prompt_len + self.num_generated

    @property
    def is_finished(self) -> bool:
        return self.num_generated >= self.max_new_tokens

    # ---------------------------------------------------------------- metrics

    @property
    def ttft(self) -> Optional[float]:
        """Time to first token: what a user perceives as responsiveness."""
        if self.first_token_at is None or self.arrived_at is None:
            return None
        return self.first_token_at - self.arrived_at

    @property
    def e2e_latency(self) -> Optional[float]:
        if self.finished_at is None or self.arrived_at is None:
            return None
        return self.finished_at - self.arrived_at

    @property
    def mean_itl(self) -> Optional[float]:
        """Mean inter-token latency: perceived streaming smoothness."""
        if len(self.token_times) < 2:
            return None
        gaps = [b - a for a, b in zip(self.token_times, self.token_times[1:])]
        return sum(gaps) / len(gaps)
