"""Correctness first: a scheduler may reorder work, never change output.

If greedy decoding through the engine does not reproduce model.generate token
for token, the batched cache is broken -- wrong slot, wrong position, or a
mask that lets one sequence attend to another's keys. Throughput numbers from
a server that computes the wrong thing are worthless, so these run first.
"""

import pytest
import torch

from minillm.config import ModelConfig
from minillm.model import MiniLLM
from minillm.serving import SCHEDULERS, InferenceEngine, Request

torch.manual_seed(0)


@pytest.fixture
def cfg():
    return ModelConfig(
        vocab_size=64, context_len=64, n_layers=2, n_heads=2,
        d_model=32, d_ff=64, dropout=0.0,
    )


@pytest.fixture
def model(cfg):
    return MiniLLM(cfg).eval()


def make_requests(n, cfg, prompt_len=6, max_new=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        Request(
            id=i,
            prompt_ids=torch.randint(0, cfg.vocab_size, (prompt_len,), generator=g).tolist(),
            max_new_tokens=max_new,
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("name", ["sequential", "static", "continuous"])
def test_matches_single_sequence_generate(name, model, cfg):
    """Every scheduler must produce exactly what model.generate produces."""
    reqs = make_requests(4, cfg)

    expected = []
    for r in reqs:
        ids = torch.tensor([r.prompt_ids], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=r.max_new_tokens, temperature=0.0)
        expected.append(out[0, len(r.prompt_ids):].tolist())

    engine = InferenceEngine(model, cfg, SCHEDULERS[name](max_batch_size=3), temperature=0.0)
    engine.run(reqs)

    for r, exp in zip(reqs, expected):
        assert r.output_ids == exp, f"{name} scheduler diverged on request {r.id}"


@pytest.mark.parametrize("name", ["sequential", "static", "continuous"])
def test_all_requests_complete(name, model, cfg):
    reqs = make_requests(6, cfg, max_new=5)
    engine = InferenceEngine(model, cfg, SCHEDULERS[name](max_batch_size=3), temperature=0.0)
    engine.run(reqs)
    assert all(r.num_generated == 5 for r in reqs)
    assert all(r.finished_at is not None for r in reqs)


def test_varied_lengths_do_not_interfere(model, cfg):
    """Different prompt and output lengths in one batch must stay independent.

    This is the case a shared cache gets wrong: sequences at different
    positions, sharing a batch, reading each other's slots.
    """
    g = torch.Generator().manual_seed(1)
    reqs = [
        Request(id=i, prompt_ids=torch.randint(0, cfg.vocab_size, (p,), generator=g).tolist(),
                max_new_tokens=m)
        for i, (p, m) in enumerate([(3, 12), (10, 4), (6, 9), (15, 2)])
    ]
    expected = []
    for r in reqs:
        ids = torch.tensor([r.prompt_ids], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=r.max_new_tokens, temperature=0.0)
        expected.append(out[0, len(r.prompt_ids):].tolist())

    engine = InferenceEngine(model, cfg, SCHEDULERS["continuous"](max_batch_size=3), temperature=0.0)
    engine.run(reqs)
    for r, exp in zip(reqs, expected):
        assert r.output_ids == exp


def test_continuous_keeps_batch_fuller_than_static(model, cfg):
    """The core claim: refilling slots as they free beats waiting for the batch."""
    g = torch.Generator().manual_seed(2)
    spec = [(5, 20), (5, 2), (5, 2), (5, 2), (5, 20), (5, 2)]

    def build():
        gg = torch.Generator().manual_seed(2)
        return [
            Request(id=i, prompt_ids=torch.randint(0, cfg.vocab_size, (p,), generator=gg).tolist(),
                    max_new_tokens=m)
            for i, (p, m) in enumerate(spec)
        ]

    stats = {}
    for name in ("static", "continuous"):
        engine = InferenceEngine(model, cfg, SCHEDULERS[name](max_batch_size=4), temperature=0.0)
        engine.run(build())
        stats[name] = engine.stats

    assert stats["continuous"].mean_batch_size > stats["static"].mean_batch_size
    assert stats["continuous"].decode_steps < stats["static"].decode_steps


def test_cache_slots_are_released(model, cfg):
    """Finished requests must return their slot, or the engine deadlocks."""
    sched = SCHEDULERS["continuous"](max_batch_size=2)
    engine = InferenceEngine(model, cfg, sched, temperature=0.0)
    engine.run(make_requests(8, cfg, max_new=3))
    assert engine.cache.num_free == 2


def test_fragmentation_is_reported(model, cfg):
    """Short requests reserve a full context window each: that is the waste
    a paged cache is designed to remove.

    Measured at peak during the run, not afterwards -- once every request
    finishes and its slot is freed, fragmentation is trivially 1.0.
    """
    reqs = make_requests(4, cfg, prompt_len=4, max_new=4)
    engine = InferenceEngine(model, cfg, SCHEDULERS["continuous"](max_batch_size=4), temperature=0.0)
    engine.run(reqs)

    # 4 slots x 64 context = 256 token capacity; 4 requests x ~8 tokens live.
    assert 0.5 < engine.stats.peak_fragmentation < 1.0
    assert engine.cache.fragmentation() == 1.0  # everything released at the end
