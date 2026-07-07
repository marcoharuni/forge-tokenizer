from forge_tokenizer.bpe import decode_tokens, encode_word, train_bpe


def test_bpe_training_deterministic_on_toy_corpus():
    freqs = {" low": 5, " lower": 2, " newest": 6, " widest": 3}
    merges_a, _ = train_bpe(freqs, 20)
    merges_b, _ = train_bpe(freqs, 20)
    assert merges_a == merges_b
    assert merges_a[0] == (b"e", b"s")


def test_bpe_dropout_reproducible_with_seed():
    freqs = {" tokenization": 10}
    merges, _ = train_bpe(freqs, 20)
    first = encode_word(" tokenization", merges, dropout=0.3, seed=123)
    second = encode_word(" tokenization", merges, dropout=0.3, seed=123)
    assert first == second


def test_decode_replaces_invalid_utf8_when_needed():
    assert decode_tokens([b"\xff"]).endswith("�")
