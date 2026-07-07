"""A compact pure Python byte-pair encoding implementation.

Training tie-break rule: at each step the pair with the highest frequency is
merged; ties are resolved lexicographically by the raw byte values of the pair.
That makes training deterministic across machines and Python versions.
"""

from __future__ import annotations

from collections import Counter
import random
from typing import Iterable


def _initial_symbols(text: str, byte_level: bool) -> tuple[bytes, ...]:
    if byte_level:
        return tuple(bytes([b]) for b in text.encode("utf-8"))
    return tuple(ch.encode("utf-8") for ch in text)


def get_pair_counts(corpus_tokens: dict[tuple[bytes, ...], int]) -> Counter:
    """Count adjacent symbol pairs in a tokenized corpus."""

    counts: Counter = Counter()
    for symbols, freq in corpus_tokens.items():
        if freq <= 0:
            continue
        for left, right in zip(symbols, symbols[1:]):
            counts[(left, right)] += freq
    return counts


def merge_pair(
    pair: tuple[bytes, bytes], corpus_tokens: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, ...], int]:
    """Return a new corpus with every adjacent occurrence of *pair* merged."""

    merged_corpus: dict[tuple[bytes, ...], int] = {}
    replacement = pair[0] + pair[1]
    for symbols, freq in corpus_tokens.items():
        out: list[bytes] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                out.append(replacement)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        merged_corpus[tuple(out)] = merged_corpus.get(tuple(out), 0) + freq
    return merged_corpus


def train_bpe(
    word_freqs: dict[str, int], num_merges: int, byte_level: bool = True
) -> tuple[list[tuple[bytes, bytes]], dict]:
    """Train BPE merges from word frequencies.

    Args:
        word_freqs: Mapping from pre-token string to integer frequency.
        num_merges: Maximum number of merge rules to learn.
        byte_level: If true, start from UTF-8 bytes. Otherwise start from
            Unicode characters encoded as UTF-8 byte strings.

    Returns:
        ``(merges, stats)`` where merges is an ordered list of byte pairs and
        stats contains the final corpus and a merge trace.
    """

    if num_merges < 0:
        raise ValueError("num_merges must be non-negative")
    corpus_tokens: dict[tuple[bytes, ...], int] = {}
    for word, freq in word_freqs.items():
        if freq <= 0:
            continue
        symbols = _initial_symbols(word, byte_level=byte_level)
        if symbols:
            corpus_tokens[symbols] = corpus_tokens.get(symbols, 0) + int(freq)

    merges: list[tuple[bytes, bytes]] = []
    trace = []
    for step in range(num_merges):
        pair_counts = get_pair_counts(corpus_tokens)
        if not pair_counts:
            break
        best_pair, best_count = min(pair_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        if best_count <= 0:
            break
        merges.append(best_pair)
        trace.append({"step": step, "pair": best_pair, "count": best_count})
        corpus_tokens = merge_pair(best_pair, corpus_tokens)

    return merges, {"final_corpus": corpus_tokens, "merge_trace": trace}


def encode_word(
    word: str,
    merges: Iterable[tuple[bytes, bytes]],
    byte_level: bool = True,
    dropout: float = 0.0,
    seed: int | None = None,
) -> list[bytes]:
    """Encode a single pre-token as BPE byte tokens.

    BPE dropout skips each merge rule with probability ``dropout`` during this
    encoding call. A seed makes the stochastic path reproducible.
    """

    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must satisfy 0 <= dropout < 1")
    symbols = list(_initial_symbols(word, byte_level=byte_level))
    rng = random.Random(seed)
    for pair in merges:
        if len(symbols) < 2:
            break
        if dropout and rng.random() < dropout:
            continue
        merged = pair[0] + pair[1]
        out: list[bytes] = []
        i = 0
        changed = False
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                out.append(merged)
                i += 2
                changed = True
            else:
                out.append(symbols[i])
                i += 1
        if changed:
            symbols = out
    return symbols


def decode_tokens(tokens: list[bytes]) -> str:
    """Decode byte tokens into text, replacing invalid UTF-8 only if needed."""

    joined = b"".join(tokens)
    try:
        return joined.decode("utf-8")
    except UnicodeDecodeError:
        return joined.decode("utf-8", errors="replace")


def build_vocab_from_merges(
    merges: Iterable[tuple[bytes, bytes]], special_tokens: Iterable[str]
) -> tuple[dict[bytes | str, int], dict[int, bytes | str]]:
    """Build a deterministic vocabulary from special tokens, bytes, and merges."""

    vocab: dict[bytes | str, int] = {}
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    for value in range(256):
        token = bytes([value])
        vocab[token] = len(vocab)
    for left, right in merges:
        token = left + right
        if token not in vocab:
            vocab[token] = len(vocab)
    id_to_token = {idx: token for token, idx in vocab.items()}
    return vocab, id_to_token
