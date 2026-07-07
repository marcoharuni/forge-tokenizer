from pathlib import Path

from forge_tokenizer import ForgeTokenizer


ROOT = Path(__file__).resolve().parents[1]


def main():
    corpus = (ROOT / "data" / "tiny_corpus.txt").read_text(encoding="utf-8")
    tokenizer = ForgeTokenizer().train(corpus, num_merges=80)
    output = ROOT / "generated" / "toy_tokenizer.json"
    output.parent.mkdir(exist_ok=True)
    tokenizer.save(output)
    print("trained toy byte-level BPE tokenizer")
    print(f"vocab_size={tokenizer.vocab_size}")
    print(f"saved={output.relative_to(ROOT)}")
    print("first_10_merges=")
    for left, right in tokenizer.merges[:10]:
        print(f"  {left!r} + {right!r} -> {left + right!r}")


if __name__ == "__main__":
    main()
