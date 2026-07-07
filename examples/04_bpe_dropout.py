from pathlib import Path

from forge_tokenizer import ForgeTokenizer
from forge_tokenizer.visualization import plot_bpe_dropout_distribution


ROOT = Path(__file__).resolve().parents[1]


def main():
    corpus = (ROOT / "data" / "tiny_corpus.txt").read_text(encoding="utf-8")
    tokenizer = ForgeTokenizer().train(corpus, num_merges=120)
    text = " tokenization tokenization tokenization"
    samples = [tokenizer.encode(text, dropout=0.2, seed=seed) for seed in range(40)]
    print("BPE dropout samples for a repeated local phrase:")
    for sample in samples[:5]:
        print(sample)
    print(f"unique_token_counts={sorted({len(sample) for sample in samples})}")
    fig = plot_bpe_dropout_distribution(samples, ROOT / "generated" / "bpe_dropout.png")
    print(f"figure={fig.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
