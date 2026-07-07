"""Unembedding, filtering, and sampling utilities."""

from __future__ import annotations

import numpy as np


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax without overflowing on large logits."""

    logits = np.asarray(logits, dtype=float)
    if logits.size == 0:
        return np.asarray([], dtype=float)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = exp.sum()
    if total == 0:
        return np.zeros_like(exp)
    return exp / total


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by temperature."""

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return np.asarray(logits, dtype=float) / temperature


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the top-k logits and set the rest to ``-inf``."""

    logits = np.asarray(logits, dtype=float)
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= logits.size:
        return logits.copy()
    filtered = np.full_like(logits, -np.inf, dtype=float)
    top_indices = np.argpartition(logits, -k)[-k:]
    filtered[top_indices] = logits[top_indices]
    return filtered


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus filter logits by keeping the smallest set with probability >= p."""

    if not 0 < p <= 1:
        raise ValueError("p must satisfy 0 < p <= 1")
    logits = np.asarray(logits, dtype=float)
    if logits.size == 0 or p == 1:
        return logits.copy()
    probs = stable_softmax(logits)
    order = np.argsort(-probs)
    cumulative = np.cumsum(probs[order])
    keep_count = int(np.searchsorted(cumulative, p, side="left")) + 1
    keep = order[:keep_count]
    filtered = np.full_like(logits, -np.inf, dtype=float)
    filtered[keep] = logits[keep]
    return filtered


def logits_to_probs(logits, temperature=1.0, top_k=None, top_p=None):
    """Apply temperature/filtering and return probabilities."""

    filtered = temperature_scale(np.asarray(logits, dtype=float), temperature)
    if top_k is not None:
        filtered = top_k_filter(filtered, top_k)
    if top_p is not None:
        filtered = top_p_filter(filtered, top_p)
    return stable_softmax(filtered)


def sample_from_logits(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> int:
    """Sample one token index from logits."""

    probs = logits_to_probs(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    if probs.size == 0:
        raise ValueError("cannot sample from empty logits")
    rng = np.random.default_rng(seed)
    return int(rng.choice(np.arange(probs.size), p=probs))
