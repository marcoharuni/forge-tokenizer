"""Tokenizer metrics used in the companion experiments."""

from __future__ import annotations

from collections import Counter
import math

import regex


def count_words(text: str) -> int:
    """Count Unicode word-like runs.

    Example:
        >>> count_words("hello, dunia")
        2
    """

    return len(regex.findall(r"\p{L}+|\p{N}+", text or ""))


def fertility(tokenizer, text: str) -> float:
    """Return tokens per word, with ``0.0`` for empty or wordless text."""

    words = count_words(text)
    if words == 0:
        return 0.0
    return len(tokenizer.encode(text)) / words


def compression_ratio(tokenizer, text: str) -> float:
    """Return UTF-8 bytes per token, with ``0.0`` for empty encodings."""

    token_count = len(tokenizer.encode(text))
    if token_count == 0:
        return 0.0
    return len(text.encode("utf-8")) / token_count


def parity_score(tokenizer, text_a: str, text_b: str) -> float:
    """Compare fertility between two texts as a symmetric ratio in ``[0, 1]``."""

    fa = fertility(tokenizer, text_a)
    fb = fertility(tokenizer, text_b)
    if fa == 0.0 and fb == 0.0:
        return 1.0
    if max(fa, fb) == 0.0:
        return 0.0
    return min(fa, fb) / max(fa, fb)


def token_frequency(tokenizer, texts: list[str]) -> dict[int, int]:
    """Count token IDs over a list of texts."""

    counts: Counter = Counter()
    for text in texts:
        counts.update(tokenizer.encode(text))
    return dict(counts)


def single_token_retention_rate(tokenizer, text: str) -> float:
    """Return the fraction of word-like runs encoded as exactly one token.

    This metric complements fertility. Fertility says how many tokens are used
    per word on average; single-token retention asks how often a word remains
    intact as one token. Empty or wordless text returns ``0.0``.

    Example:
        >>> from forge_tokenizer import ForgeTokenizer
        >>> tok = ForgeTokenizer().train("hello world hello", num_merges=20)
        >>> 0.0 <= single_token_retention_rate(tok, "hello world") <= 1.0
        True
    """

    words = regex.findall(r"\p{L}+|\p{N}+", text or "")
    if not words:
        return 0.0
    retained = sum(1 for word in words if len(tokenizer.encode(word)) == 1)
    return retained / len(words)


def renyi_entropy(probs: list[float], alpha: float = 2.5) -> float:
    """Compute Renyi entropy for a probability vector.

    Zero probabilities are ignored. For ``alpha=1`` this returns Shannon
    entropy by continuity.
    """

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    cleaned = [float(p) for p in probs if p > 0]
    total = sum(cleaned)
    if total <= 0:
        return 0.0
    normalized = [p / total for p in cleaned]
    if math.isclose(alpha, 1.0):
        return -sum(p * math.log(p) for p in normalized)
    return math.log(sum(p**alpha for p in normalized)) / (1.0 - alpha)


def renyi_efficiency(token_counts: dict[int, int], alpha: float = 2.5) -> float:
    """Return entropy divided by maximum entropy for the observed support."""

    counts = [count for count in token_counts.values() if count > 0]
    if len(counts) <= 1:
        return 0.0
    entropy = renyi_entropy(counts, alpha=alpha)
    return entropy / math.log(len(counts))


def roundtrip_accuracy(tokenizer, texts: list[str]) -> float:
    """Fraction of texts that decode exactly after encoding."""

    if not texts:
        return 0.0
    correct = sum(1 for text in texts if tokenizer.decode(tokenizer.encode(text)) == tokenizer._prepare(text))
    return correct / len(texts)


def formatting_overhead(tokenizer, code: str) -> float:
    """Estimate the token cost of formatting by comparing code to collapsed text."""

    formatted = len(tokenizer.encode(code))
    collapsed_text = regex.sub(r"\s+", " ", code).strip()
    collapsed = len(tokenizer.encode(collapsed_text))
    if collapsed == 0:
        return 0.0
    return (formatted - collapsed) / collapsed
