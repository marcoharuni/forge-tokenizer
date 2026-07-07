from forge_tokenizer import ForgeTokenizer
from forge_tokenizer.metrics import (
    compression_ratio,
    fertility,
    formatting_overhead,
    parity_score,
    renyi_efficiency,
    renyi_entropy,
    roundtrip_accuracy,
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


def test_roundtrip_accuracy_nonempty():
    tokenizer = ForgeTokenizer().train("hello world", num_merges=10)
    assert roundtrip_accuracy(tokenizer, ["hello", "world"]) == 1.0
