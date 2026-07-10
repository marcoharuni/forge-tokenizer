# forge-tokenizer

This repository is an educational reference implementation. It is designed for
clarity and reproducibility, not maximum tokenizer throughput.

## What This Repo Teaches

- Unicode normalization and code point inspection
- GPT-style regex pretokenization
- Byte-level BPE training from first principles
- Encode/decode round trips with stable special tokens
- Tokenizer metrics such as fertility, compression, parity, and formatting overhead
- Single-token retention rate as a simple complement to fertility
- Embedding geometry with one-hot vectors, norms, PPMI, SVD, neighbors, and analogies
- Unembedding mechanics with stable softmax, temperature, top-k, top-p, and sampling
- Chat-style formatting token overhead
- A simulated logit lens demonstration
- Local-only benchmark discipline

## What This Repo Is Not

This is not a production replacement for `tiktoken`, HuggingFace Tokenizers, or
SentencePiece. The core implementation intentionally avoids external tokenizer
libraries, Rust, C++, CUDA, Triton, Torch, and Transformers.

## Installation

Fresh local setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

That installs the package and the lightweight dependencies used by the examples
and tests: `numpy`, `matplotlib`, `pytest`, `tqdm`, and `regex`.

Verify the install:

```bash
pytest -q
python examples/01_train_toy_bpe.py
python examples/05_ppmi_svd_embeddings.py
python examples/07_logit_lens_demo.py
```

## Open In Colab

You can run the learning notebooks directly in Google Colab. Open a notebook,
run the first **Install & Clone** cell, then continue from top to bottom.

[![Open Tokenizer Playground in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoharuni/forge-tokenizer/blob/main/notebooks/tokenizer_playground.ipynb)

[![Open Embedding Geometry in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoharuni/forge-tokenizer/blob/main/notebooks/embedding_geometry.ipynb)

[![Open BPE Merge Trace in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoharuni/forge-tokenizer/blob/main/notebooks/03_bpe_merge_trace.ipynb)

[![Open Token Tax Lab in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoharuni/forge-tokenizer/blob/main/notebooks/04_token_tax_lab.ipynb)

[![Open Token ID To Meaning in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoharuni/forge-tokenizer/blob/main/notebooks/05_token_id_to_meaning.ipynb)

[![Open Chat Formatting Token Tax in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoharuni/forge-tokenizer/blob/main/notebooks/06_chat_formatting_token_tax.ipynb)

Each notebook contains a setup cell that clones this repository inside Colab,
installs it with `pip install -e .`, and adds the local `src/` directory to the
notebook path.

Recommended learning path:

1. `notebooks/tokenizer_playground.ipynb`
2. `notebooks/03_bpe_merge_trace.ipynb`
3. `notebooks/04_token_tax_lab.ipynb`
4. `notebooks/embedding_geometry.ipynb`
5. `notebooks/05_token_id_to_meaning.ipynb`
6. `notebooks/06_chat_formatting_token_tax.ipynb`

## Quickstart

```python
from forge_tokenizer import ForgeTokenizer

tokenizer = ForgeTokenizer().train("hello world hello", num_merges=10)
ids = tokenizer.encode("hello world")
text = tokenizer.decode(ids)
print(ids)
print(text)
```

## Train A Tokenizer

```bash
python -m forge_tokenizer.cli train --input data/tiny_corpus.txt --output toy_tokenizer.json --merges 80
```

## Encode And Decode

```bash
python -m forge_tokenizer.cli encode --tokenizer toy_tokenizer.json --text "hello world"
python -m forge_tokenizer.cli decode --tokenizer toy_tokenizer.json --ids "1,2,3"
```

## Measure Fertility

```bash
python examples/03_measure_fertility.py
```

The numbers are local measurements from the included tiny corpus and are not
universal tokenizer claims.

## Build Embeddings

```bash
python examples/05_ppmi_svd_embeddings.py
python examples/06_embedding_neighbors.py
```

These examples build small PPMI/SVD embeddings from `data/tiny_corpus.txt`.

## Run Logit Lens Simulation

```bash
python examples/07_logit_lens_demo.py
```

The logit lens demo is simulated with random matrices. It is not a trained model.

## Run Tests

```bash
pytest -q
```

## Run Benchmarks

```bash
python benchmarks/benchmark_encode.py
python benchmarks/benchmark_fertility.py
```

Benchmarks print `LOCAL MEASUREMENT ONLY` and measure the current local machine.


## Citation

If you use this repository, cite:

```bibtex
@software{haruni2026forgetokenizer,
  title={forge-tokenizer},
  author={Haruni, Marco},
  year={2026},
  url={https://github.com/marcoharuni/forge-tokenizer},
  license={MIT}
}
```

See `CITATION.cff` for citation metadata.

## License

MIT License. See `LICENSE`.
