import numpy as np

from forge_tokenizer.embeddings import (
    build_cooccurrence,
    cosine_similarity,
    cosine_similarity_matrix,
    nearest_neighbors,
    one_hot,
    ppmi_matrix,
    svd_embeddings,
)


def test_cosine_similarity_identical_vector_near_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) > 0.999999


def test_cosine_similarity_zero_vector_case_returns_zero():
    assert cosine_similarity([0, 0], [1, 2]) == 0.0


def test_one_hot_and_ppmi_svd_neighbors():
    assert np.allclose(one_hot(1, 3), [0, 1, 0])
    cooc, word_to_idx, idx_to_word = build_cooccurrence([["king", "queen", "crown"], ["king", "crown"]])
    ppmi = ppmi_matrix(cooc)
    emb = svd_embeddings(ppmi, dim=2)
    assert emb.shape == (3, 2)
    neighbors = nearest_neighbors(emb, word_to_idx, idx_to_word, "king", k=1)
    assert len(neighbors) == 1


def test_cosine_similarity_matrix_zero_safe():
    result = cosine_similarity_matrix(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]))
    assert result[0, 0] == 0.0
