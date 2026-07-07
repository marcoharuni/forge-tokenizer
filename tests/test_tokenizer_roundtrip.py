from pathlib import Path

from forge_tokenizer import ForgeTokenizer


SAMPLES = [
    "hello world",
    "Habari dunia na asante sana",
    "emoji 🔥🚀 works",
    "def add(a, b):\n    return a + b",
    "mixed café, Swahili, numbers 123, and symbols != ==",
]


def build_tokenizer():
    corpus = "\n".join(SAMPLES) + "\nTokenization embeddings language models."
    return ForgeTokenizer().train(corpus, num_merges=100)


def test_roundtrip_for_ordinary_unicode_strings():
    tokenizer = build_tokenizer()
    for sample in SAMPLES:
        assert tokenizer.decode(tokenizer.encode(sample)) == sample


def test_save_load_preserves_output_exactly(tmp_path: Path):
    tokenizer = build_tokenizer()
    text = SAMPLES[-1]
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = ForgeTokenizer.load(path)
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_special_tokens_have_stable_ids():
    tokenizer = ForgeTokenizer()
    assert tokenizer.token_to_id("<pad>") == 0
    assert tokenizer.token_to_id("<bos>") == 1
    assert tokenizer.token_to_id("<eos>") == 2
    assert tokenizer.token_to_id("<unk>") == 3


def test_explain_returns_token_id_and_decoded_piece():
    tokenizer = build_tokenizer()
    row = tokenizer.explain("hello")[0]
    assert {"id", "token", "decoded"} <= set(row)
