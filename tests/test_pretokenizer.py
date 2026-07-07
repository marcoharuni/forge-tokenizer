from forge_tokenizer.pretokenizer import pretokenize, pretokenize_with_offsets, space_aware_pretokenize


def test_pretokenizer_preserves_leading_spaces():
    tokens = pretokenize(" hello world")
    assert tokens[0] == " hello"
    assert "".join(tokens) == " hello world"


def test_offsets_reconstruct_text():
    text = "Hi, dunia!\n  café"
    rows = pretokenize_with_offsets(text)
    assert "".join(token for token, _, _ in rows) == text
    assert text[rows[-1][1] : rows[-1][2]] == rows[-1][0]


def test_space_aware_keeps_spaces():
    assert space_aware_pretokenize("a  b") == ["a", "  ", "b"]
