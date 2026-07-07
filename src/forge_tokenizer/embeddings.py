"""NumPy-only embedding geometry utilities."""

from __future__ import annotations

import numpy as np


def one_hot(index: int, vocab_size: int) -> np.ndarray:
    """Return a one-hot vector."""

    if not 0 <= index < vocab_size:
        raise ValueError("index must be inside the vocabulary")
    vec = np.zeros(vocab_size, dtype=float)
    vec[index] = 1.0
    return vec


def embedding_parameter_count(vocab_size: int, d_model: int, tied: bool = False) -> dict:
    """Count embedding and unembedding parameters."""

    embedding = vocab_size * d_model
    unembedding = 0 if tied else vocab_size * d_model
    return {"embedding": embedding, "unembedding": unembedding, "total": embedding + unembedding, "tied": tied}


def cosine_similarity(u, v) -> float:
    """Cosine similarity, returning ``0.0`` when either vector is zero."""

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0:
        return 0.0
    return float(np.dot(u, v) / denom)


def cosine_similarity_matrix(A, B) -> np.ndarray:
    """Pairwise cosine similarity between rows of ``A`` and rows of ``B``."""

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)
    A_safe = np.divide(A, A_norm, out=np.zeros_like(A), where=A_norm != 0)
    B_safe = np.divide(B, B_norm, out=np.zeros_like(B), where=B_norm != 0)
    return A_safe @ B_safe.T


def l1_norm(v):
    return float(np.linalg.norm(v, ord=1))


def l2_norm(v):
    return float(np.linalg.norm(v, ord=2))


def linf_norm(v):
    return float(np.linalg.norm(v, ord=np.inf))


def frobenius_norm(M):
    return float(np.linalg.norm(M, ord="fro"))


def build_cooccurrence(sentences: list[list[str]], window_size: int = 2):
    """Build a symmetric word co-occurrence matrix.

    Returns ``(matrix, word_to_idx, idx_to_word)`` so examples can keep the
    matrix and vocabulary aligned without global state.
    """

    if window_size < 1:
        raise ValueError("window_size must be positive")
    vocab = sorted({word for sentence in sentences for word in sentence})
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    cooc = np.zeros((len(vocab), len(vocab)), dtype=float)
    for sentence in sentences:
        for i, word in enumerate(sentence):
            if word not in word_to_idx:
                continue
            left = max(0, i - window_size)
            right = min(len(sentence), i + window_size + 1)
            for j in range(left, right):
                if i == j:
                    continue
                cooc[word_to_idx[word], word_to_idx[sentence[j]]] += 1.0
    return cooc, word_to_idx, idx_to_word


def ppmi_matrix(cooc: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert co-occurrence counts to positive PMI values."""

    cooc = np.asarray(cooc, dtype=float)
    total = cooc.sum()
    if total <= 0:
        return np.zeros_like(cooc)
    row_sum = cooc.sum(axis=1, keepdims=True)
    col_sum = cooc.sum(axis=0, keepdims=True)
    expected = (row_sum @ col_sum) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((cooc + eps) / (expected + eps))
    return np.maximum(pmi, 0.0)


def svd_embeddings(ppmi: np.ndarray, dim: int = 10) -> np.ndarray:
    """Build low-dimensional embeddings from a PPMI matrix using SVD."""

    if dim < 1:
        raise ValueError("dim must be positive")
    if ppmi.size == 0:
        return np.zeros((0, dim))
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    used = min(dim, U.shape[1])
    out = U[:, :used] * np.sqrt(S[:used])
    if used < dim:
        out = np.pad(out, ((0, 0), (0, dim - used)))
    return out


def nearest_neighbors(embeddings, word_to_idx, idx_to_word, query_word, k=10):
    """Return nearest words by cosine similarity."""

    if query_word not in word_to_idx:
        raise KeyError(f"unknown word: {query_word}")
    matrix = np.asarray(embeddings, dtype=float)
    query_idx = word_to_idx[query_word]
    scores = cosine_similarity_matrix(matrix[[query_idx]], matrix)[0]
    order = np.argsort(-scores)
    results = []
    for idx in order:
        word = idx_to_word[int(idx)]
        if word == query_word:
            continue
        results.append((word, float(scores[idx])))
        if len(results) >= k:
            break
    return results


def analogy(embeddings, word_to_idx, idx_to_word, a, b, c, k=10):
    """Solve ``a is to b as c is to ?`` using vector arithmetic."""

    for word in (a, b, c):
        if word not in word_to_idx:
            raise KeyError(f"unknown word: {word}")
    matrix = np.asarray(embeddings, dtype=float)
    target = matrix[word_to_idx[b]] - matrix[word_to_idx[a]] + matrix[word_to_idx[c]]
    scores = cosine_similarity_matrix(np.asarray([target]), matrix)[0]
    blocked = {word_to_idx[a], word_to_idx[b], word_to_idx[c]}
    results = []
    for idx in np.argsort(-scores):
        if int(idx) in blocked:
            continue
        results.append((idx_to_word[int(idx)], float(scores[idx])))
        if len(results) >= k:
            break
    return results
