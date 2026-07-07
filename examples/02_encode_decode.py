from pathlib import Path

from forge_tokenizer import ForgeTokenizer


ROOT = Path(__file__).resolve().parents[1]


def build_tokenizer():
    corpus = (ROOT / "data" / "tiny_corpus.txt").read_text(encoding="utf-8")
    return ForgeTokenizer().train(corpus, num_merges=80)


def main():
    tokenizer = build_tokenizer()
    text = "Habari dunia, tokenizer sees café and 🔥."
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print(f"text={text}")
    print(f"ids={ids}")
    print(f"decoded={decoded}")
    print(f"roundtrip={decoded == text}")
    print("explain_first_8=")
    for row in tokenizer.explain(text)[:8]:
        print(row)


if __name__ == "__main__":
    main()
