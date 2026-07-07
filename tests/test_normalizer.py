import pytest

from forge_tokenizer.normalizer import explain_codepoints, normalize_text, validate_unicode


def test_unicode_normalization_equivalence_for_e_acute():
    composed = "é"
    decomposed = "e\u0301"
    assert normalize_text(composed, "NFC") == normalize_text(decomposed, "NFC")


def test_unsupported_normalization_form_raises():
    with pytest.raises(ValueError):
        normalize_text("hello", "BAD")


def test_explain_codepoints_contains_names():
    rows = explain_codepoints("A")
    assert rows[0]["codepoint"] == "U+0041"
    assert "LATIN" in rows[0]["name"]


def test_validate_unicode_rejects_surrogate():
    with pytest.raises(ValueError):
        validate_unicode("\ud800")
