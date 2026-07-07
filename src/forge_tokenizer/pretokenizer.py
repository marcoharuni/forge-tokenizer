"""Small Unicode-aware pretokenizers.

The default pattern is a simplified GPT-style expression. It keeps a leading
space attached to the following word, number, or punctuation run when possible,
which makes whitespace visible to a downstream byte-level BPE tokenizer.
"""

from __future__ import annotations

import regex

GPT_STYLE_PATTERN = regex.compile(
    r"""(?ix)
    '(?:[sdmt]|ll|ve|re)
    | [ ]?\p{L}+
    | [ ]?\p{N}+
    | [ ]?[^\s\p{L}\p{N}]+
    |\s+(?!\S)
    |\s+
    """
)


def pretokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Split text into regex pre-tokens and return ``(token, start, end)`` rows.

    Offsets are Python string offsets. The function does not strip or normalize
    input; every character is represented in order.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a str")
    return [(match.group(0), match.start(), match.end()) for match in GPT_STYLE_PATTERN.finditer(text)]


def pretokenize(text: str) -> list[str]:
    """Split text with the simplified GPT-style Unicode regex pattern."""

    return [token for token, _, _ in pretokenize_with_offsets(text)]


def simple_whitespace_pretokenize(text: str) -> list[str]:
    """Split on whitespace for comparison without preserving separators."""

    if not isinstance(text, str):
        raise TypeError("text must be a str")
    return text.split()


def space_aware_pretokenize(text: str) -> list[str]:
    """Split into whitespace and non-whitespace runs without dropping spaces."""

    if not isinstance(text, str):
        raise TypeError("text must be a str")
    return regex.findall(r"\s+|\S+", text)
