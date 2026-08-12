import pytest
import torch

from minillm.config import ModelConfig
from minillm.model import MiniLLM

torch.manual_seed(0)


@pytest.fixture
def cfg():
    return ModelConfig(
        vocab_size=64, context_len=32, n_layers=2, n_heads=2,
        d_model=32, d_ff=64, dropout=0.0,
    )


def test_forward_shapes(cfg):
    model = MiniLLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0


def test_weight_tying(cfg):
    model = MiniLLM(cfg)
    assert model.lm_head.weight is model.embed.weight


def test_causality(cfg):
    """Changing a later token must not alter logits at earlier positions."""
    model = MiniLLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        base, _ = model(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size
        changed, _ = model(x2)
    assert torch.allclose(base[:, :-1], changed[:, :-1], atol=1e-5)


def test_sdpa_matches_manual_attention(cfg):
    """The fused and hand-written attention paths must agree."""
    x = torch.randint(0, cfg.vocab_size, (2, 16))

    cfg.use_sdpa = True
    fused = MiniLLM(cfg).eval()
    cfg.use_sdpa = False
    manual = MiniLLM(cfg).eval()
    manual.load_state_dict(fused.state_dict())

    with torch.no_grad():
        a, _ = fused(x)
        b, _ = manual(x)
    assert torch.allclose(a, b, atol=1e-5)


def test_kv_cache_matches_full_forward(cfg):
    """Incremental decoding with the cache must produce identical logits to
    recomputing the whole prefix. This is the test that catches RoPE offset
    and causal-mask bugs -- both fail silently otherwise."""
    model = MiniLLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 12))

    with torch.no_grad():
        full, _ = model(x)

        model.clear_kv_cache()
        prefill, _ = model(x[:, :8], use_cache=True, pos_offset=0)
        assert torch.allclose(prefill, full[:, :8], atol=1e-5)

        logits = [prefill[:, -1]]
        for i in range(8, 12):
            step, _ = model(x[:, i : i + 1], use_cache=True, pos_offset=i)
            logits.append(step[:, -1])
        model.clear_kv_cache()

    stacked = torch.stack(logits[1:], dim=1)
    assert torch.allclose(stacked, full[:, 8:12], atol=1e-5)


def test_greedy_generation_matches_with_and_without_cache(cfg):
    model = MiniLLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        cached = model.generate(prompt, max_new_tokens=10, temperature=0.0, use_cache=True)
        naive = model.generate(prompt, max_new_tokens=10, temperature=0.0, use_cache=False)
    assert torch.equal(cached, naive)


def test_generate_respects_context_limit(cfg):
    model = MiniLLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(prompt, max_new_tokens=1000, temperature=0.0)
    assert out.shape[1] <= cfg.context_len
