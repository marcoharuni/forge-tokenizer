from pathlib import Path

import regex

from forge_tokenizer.embeddings import analogy, build_cooccurrence, nearest_neighbors, ppmi_matrix, svd_embeddings


ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT / "data" / "tiny_corpus.txt").read_text(encoding="utf-8").lower()
    sentences = [regex.findall(r"\p{L}+", line) for line in text.splitlines()]
    sentences = [sentence for sentence in sentences if sentence]
    cooc, word_to_idx, idx_to_word = build_cooccurrence(sentences, window_size=3)
    embeddings = svd_embeddings(ppmi_matrix(cooc), dim=8)
    query = "embeddings" if "embeddings" in word_to_idx else next(iter(word_to_idx))
    print(f"nearest neighbors for {query}:")
    for word, score in nearest_neighbors(embeddings, word_to_idx, idx_to_word, query, k=6):
        print(f"  {word}: {score:.3f}")
    candidates = ["text", "numbers", "tokens"]
    if all(word in word_to_idx for word in candidates):
        print("toy analogy: text is to numbers as tokens is to ?")
        print(analogy(embeddings, word_to_idx, idx_to_word, "text", "numbers", "tokens", k=5))
    else:
        print("toy analogy skipped because the tiny vocabulary lacks one required word")


if __name__ == "__main__":
    main()
