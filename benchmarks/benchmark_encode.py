from pathlib import Path
import time

from forge_tokenizer import ForgeTokenizer


ROOT = Path(__file__).resolve().parents[1]


def main():
    print("LOCAL MEASUREMENT ONLY — not a production tokenizer benchmark.")
    corpus = (ROOT / "data" / "tiny_corpus.txt").read_text(encoding="utf-8")
    tokenizer = ForgeTokenizer().train(corpus, num_merges=100)
    text = corpus * 200
    start = time.perf_counter()
    ids = tokenizer.encode(text)
    elapsed = time.perf_counter() - start
    chars_per_second = len(text) / elapsed if elapsed else 0.0
    print(f"characters={len(text)}")
    print(f"tokens={len(ids)}")
    print(f"seconds={elapsed:.6f}")
    print(f"characters_per_second={chars_per_second:.2f}")


if __name__ == "__main__":
    main()
