from pathlib import Path

import regex

from forge_tokenizer.embeddings import build_cooccurrence, nearest_neighbors, ppmi_matrix, svd_embeddings
from forge_tokenizer.visualization import plot_embedding_neighbors_2d


ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT / "data" / "tiny_corpus.txt").read_text(encoding="utf-8").lower()
    sentences = []
    for line in text.splitlines():
        words = regex.findall(r"\p{L}+", line)
        if words:
            sentences.append(words)
    cooc, word_to_idx, idx_to_word = build_cooccurrence(sentences, window_size=2)
    ppmi = ppmi_matrix(cooc)
    embeddings = svd_embeddings(ppmi, dim=2)
    query = "tokenization" if "tokenization" in word_to_idx else next(iter(word_to_idx))
    print(f"vocab_size={len(word_to_idx)}")
    print(f"query={query}")
    print("neighbors=")
    for word, score in nearest_neighbors(embeddings, word_to_idx, idx_to_word, query, k=8):
        print(f"  {word}: {score:.3f}")
    labels = [idx_to_word[i] for i in range(min(20, len(idx_to_word)))]
    fig = plot_embedding_neighbors_2d(embeddings[: len(labels)], labels, ROOT / "generated" / "embedding_neighbors.png")
    print(f"figure={fig.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
