import numpy as np
import pytest

from forge_tokenizer.unembedding import logits_to_probs, sample_from_logits, stable_softmax, top_k_filter, top_p_filter


def test_stable_softmax_handles_large_logits():
    probs = stable_softmax(np.array([1000, 1001, 1002]))
    assert np.isfinite(probs).all()
    assert np.isclose(probs.sum(), 1.0)
    assert probs.argmax() == 2


def test_top_k_behaves_correctly():
    filtered = top_k_filter(np.array([1.0, 3.0, 2.0]), k=2)
    assert np.isneginf(filtered[0])
    assert filtered[1] == 3.0
    assert filtered[2] == 2.0


def test_top_p_behaves_correctly():
    logits = np.array([5.0, 4.0, 1.0, 0.0])
    filtered = top_p_filter(logits, p=0.8)
    assert filtered[0] == 5.0
    assert filtered[1] == 4.0
    assert np.isneginf(filtered[2])


def test_top_p_validation_and_sampling():
    with pytest.raises(ValueError):
        top_p_filter(np.array([1.0]), p=0.0)
    probs = logits_to_probs([1.0, 2.0], top_k=1)
    assert np.argmax(probs) == 1
    assert sample_from_logits(np.array([0.0, 10.0]), seed=0) == 1
