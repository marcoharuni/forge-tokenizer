import json
from pathlib import Path

from forge_tokenizer import ForgeTokenizer
from forge_tokenizer.metrics import fertility


ROOT = Path(__file__).resolve().parents[1]


def main():
    print("LOCAL MEASUREMENT ONLY — fertility depends on this tiny local corpus and is not universal.")
    corpus = (ROOT / "data" / "tiny_corpus.txt").read_text(encoding="utf-8")
    samples = json.loads((ROOT / "data" / "multilingual_samples.json").read_text(encoding="utf-8"))
    tokenizer = ForgeTokenizer().train(corpus, num_merges=100)
    for language, text in samples.items():
        print(f"{language}: {fertility(tokenizer, text):.4f}")


if __name__ == "__main__":
    main()
