from forge_tokenizer import ForgeTokenizer
from forge_tokenizer.metrics import (
    compression_ratio,
    fertility,
    formatting_overhead,
    parity_score,
    renyi_efficiency,
    renyi_entropy,
    roundtrip_accuracy,
    single_token_retention_rate,
    token_frequency,
)


def test_metrics_handle_empty_input():
    tokenizer = ForgeTokenizer().train("hello world", num_merges=5)
    assert fertility(tokenizer, "") == 0.0
    assert compression_ratio(tokenizer, "") == 0.0
    assert parity_score(tokenizer, "", "") == 1.0
    assert token_frequency(tokenizer, []) == {}
    assert renyi_entropy([]) == 0.0
    assert renyi_efficiency({}) == 0.0
    assert roundtrip_accuracy(tokenizer, []) == 0.0
    assert formatting_overhead(tokenizer, "") == 0.0
    assert single_token_retention_rate(tokenizer, "") == 0.0


def test_roundtrip_accuracy_nonempty():
    tokenizer = ForgeTokenizer().train("hello world", num_merges=10)
    assert roundtrip_accuracy(tokenizer, ["hello", "world"]) == 1.0


def test_single_token_retention_rate_is_bounded():
    tokenizer = ForgeTokenizer().train("hello world hello", num_merges=20)
    value = single_token_retention_rate(tokenizer, "hello unknown world")
    assert 0.0 <= value <= 1.0
