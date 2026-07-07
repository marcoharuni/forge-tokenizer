# Contributing

This repository is an educational reference implementation. Contributions should
preserve clarity, determinism, and small dependency footprint.

## Local Checks

Run these before opening a pull request:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -q
python examples/01_train_toy_bpe.py
python examples/05_ppmi_svd_embeddings.py
python examples/07_logit_lens_demo.py
python benchmarks/benchmark_encode.py
```

Benchmark output must be reported as a local measurement only. Do not add
precomputed performance claims.

## Implementation Principles

- Keep the core tokenizer implementation pure Python.
- Do not add tokenizer libraries to the core implementation.
- Prefer small, readable examples over production abstractions.
- Keep tests independent of internet downloads.
- Mark optional comparison code clearly if it uses external tokenizers.
