"""Schedulers.

All three answer one question each iteration: which requests run now? That is
the only thing that differs between them -- the model, the cache, and the
decode kernel are identical. Any throughput difference is a scheduling result,
not a kernel one, which is what makes the comparison meaningful.

    Sequential   one request start to finish, then the next
    Static       fill a batch, run it until every member finishes
    Continuous   re-decide membership every decode step
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .request import Request, State


class Scheduler(ABC):
    """Owns the waiting queue and decides batch membership."""

    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self.waiting: list[Request] = []
        self.running: list[Request] = []

    def add(self, request: Request) -> None:
        self.waiting.append(request)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @abstractmethod
    def schedule(self) -> tuple[list[Request], list[Request]]:
        """Return (to_admit, to_evict) for this iteration."""

    def _finished(self) -> list[Request]:
        return [r for r in self.running if r.is_finished]


class SequentialScheduler(Scheduler):
    """Baseline: batch size one, no concurrency.

    Every request waits for all requests ahead of it to finish completely.
    The GPU spends the entire run decoding a single sequence, so arithmetic
    intensity is dismal -- weights are loaded from memory to serve one token.
    This exists to be beaten.
    """

    def __init__(self, max_batch_size: int = 1):
        super().__init__(max_batch_size=1)

    def schedule(self):
        evict = self._finished()
        for r in evict:
            self.running.remove(r)

        admit = []
        if not self.running and self.waiting:
            admit.append(self.waiting.pop(0))
        return admit, evict


class StaticBatchScheduler(Scheduler):
    """Fill a batch, then run it to completion before touching the queue.

    Better than sequential -- the batch amortizes weight loading across
    sequences -- but it has two structural problems that the benchmark makes
    visible:

    Head-of-line blocking: the batch is only released when its *longest*
    member finishes. A request generating 20 tokens alongside one generating
    200 occupies its slot, doing nothing, for 180 wasted steps.

    Admission stalling: requests arriving mid-batch wait for the whole batch
    regardless of how much capacity is idle.
    """

    def schedule(self):
        # A batch in flight is untouchable until every member is done.
        if self.running:
            if not all(r.is_finished for r in self.running):
                return [], []
            evict = list(self.running)
            self.running.clear()
            return [], evict

        n = min(self.max_batch_size, len(self.waiting))
        admit = [self.waiting.pop(0) for _ in range(n)]
        return admit, []


class ContinuousBatchScheduler(Scheduler):
    """Iteration-level scheduling: membership is reconsidered every step.

    A finished sequence is evicted immediately and its cache slot is refilled
    from the queue on the very next decode step, so the batch stays as full as
    the queue allows instead of draining down to its slowest member. This is
    the idea behind Orca and vLLM, and it is where the throughput gain lives.

    The cost is that admission now involves a prefill, which is a much larger
    forward pass than a decode step. Under a heavy arrival rate prefills
    interleave with decodes and can visibly stall token streaming for the
    sequences already running -- the trade that chunked prefill exists to fix.
    """

    def schedule(self):
        evict = self._finished()
        for r in evict:
            self.running.remove(r)

        free_slots = self.max_batch_size - len(self.running)
        n = min(free_slots, len(self.waiting))
        admit = [self.waiting.pop(0) for _ in range(n)]
        return admit, evict


SCHEDULERS = {
    "sequential": SequentialScheduler,
    "static": StaticBatchScheduler,
    "continuous": ContinuousBatchScheduler,
}
