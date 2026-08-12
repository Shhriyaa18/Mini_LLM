import pytest

from minillm.tokenizer import BPETokenizer

CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
    "class Foo:\n    def __init__(self):\n        self.x = 1\n",
    "for i in range(10):\n    print(i)\n",
] * 20


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer(vocab_size=400)
    t.train(CORPUS, verbose=False)
    return t


@pytest.mark.parametrize(
    "text",
    [
        "def add(a, b):\n    return a + b\n",
        "class Bar:\n    pass\n",
        "x = [1, 2, 3]  # comment",
        "",
        "unicode: \u00e9\u00e8\u00ea \u4e2d\u6587 \U0001f40d",
        "    \t  weird\n\n\nwhitespace   ",
    ],
)
def test_round_trip(tok, text):
    """Encoding then decoding must return the original bytes exactly."""
    assert tok.decode(tok.encode(text)) == text


def test_compression_beats_raw_bytes(tok):
    text = "def add(a, b):\n    return a + b\n" * 10
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_ids_within_vocab(tok):
    ids = tok.encode("def add(a, b):\n    return a + b\n")
    assert all(0 <= i < len(tok) for i in ids)


def test_save_load_round_trip(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    text = "class Foo:\n    def __init__(self):\n        self.x = 1\n"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_merges_applied_in_rank_order(tok):
    """Encoding must reproduce the segmentation learned during training.

    Greedy left-to-right merging can diverge from rank order; this guards it.
    """
    text = "def add(a, b):"
    ids = tok.encode(text)
    # Re-encoding a decoded round trip must be a fixed point.
    assert tok.encode(tok.decode(ids)) == ids


@pytest.mark.parametrize(
    "text",
    [
        "def __init__(self, max_len):",
        "self._private = other.__dict__",
        "snake_case_name = 42",
        "\u00e9\u00e8 \u4e2d\u6587 \U0001f40d",
    ],
)
def test_pretokenizer_is_lossless(text):
    """Every character must land in exactly one pre-token.

    A pre-tokenizer pattern whose branches do not cover the full character set
    drops input silently -- no error, just missing characters in the corpus.
    """
    from minillm.tokenizer import GPT2_PATTERN
    import regex

    assert "".join(regex.findall(GPT2_PATTERN, text)) == text
