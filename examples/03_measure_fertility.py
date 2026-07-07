import json
from pathlib import Path

from forge_tokenizer import ForgeTokenizer
from forge_tokenizer.metrics import compression_ratio, fertility
from forge_tokenizer.visualization import plot_fertility_bar


ROOT = Path(__file__).resolve().parents[1]


def main():
    corpus = (ROOT / "data" / "tiny_corpus.txt").read_text(encoding="utf-8")
    samples = json.loads((ROOT / "data" / "multilingual_samples.json").read_text(encoding="utf-8"))
    tokenizer = ForgeTokenizer().train(corpus, num_merges=80)
    results = {language: fertility(tokenizer, text) for language, text in samples.items()}
    print("Local fertility measurements from the tiny included corpus:")
    for language, value in results.items():
        print(f"{language}: fertility={value:.3f}, compression={compression_ratio(tokenizer, samples[language]):.3f}")
    fig = plot_fertility_bar(results, ROOT / "generated" / "fertility.png")
    print(f"figure={fig.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
