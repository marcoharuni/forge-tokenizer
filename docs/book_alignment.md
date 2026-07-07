# Book Alignment

| Repository file | Book support |
| --- | --- |
| `src/forge_tokenizer/normalizer.py` | Unicode and text normalization foundations |
| `src/forge_tokenizer/pretokenizer.py` | Pretokenization and GPT-style regex splitting |
| `src/forge_tokenizer/bpe.py` | Byte-level BPE from first principles |
| `src/forge_tokenizer/tokenizer.py` | End-to-end tokenizer API, special tokens, save/load |
| `src/forge_tokenizer/metrics.py` | Fertility, token tax, parity, round-trip checks |
| `src/forge_tokenizer/embeddings.py` | One-hot vectors, norms, PPMI, SVD, neighbors |
| `src/forge_tokenizer/unembedding.py` | Softmax, temperature, top-k, top-p, sampling |
| `src/forge_tokenizer/logit_lens.py` | Simulated logit lens mechanism |
| `src/forge_tokenizer/visualization.py` | Reproducible book figures |
| `src/forge_tokenizer/cli.py` | Command-line workflow for readers |
| `examples/01_train_toy_bpe.py` | BPE training chapter |
| `examples/02_encode_decode.py` | Encoding, decoding, and explainability |
| `examples/03_measure_fertility.py` | Multilingual fertility and token tax |
| `examples/04_bpe_dropout.py` | Stochastic segmentation |
| `examples/05_ppmi_svd_embeddings.py` | Distributional embeddings |
| `examples/06_embedding_neighbors.py` | Embedding geometry and analogies |
| `examples/07_logit_lens_demo.py` | Unembedding and logit lens intuition |
| `benchmarks/` | Local measurement discipline |
| `data/` | Small reproducible corpora and samples |
| `tests/` | Regression coverage for reader-facing behavior |
