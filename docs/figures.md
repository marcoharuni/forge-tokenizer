# Figures

Generated figures are written to `generated/`, which is ignored by Git.

Regenerate common figures with:

```bash
python examples/03_measure_fertility.py
python examples/04_bpe_dropout.py
python examples/05_ppmi_svd_embeddings.py
python examples/07_logit_lens_demo.py
```

The plotting functions live in `src/forge_tokenizer/visualization.py` and use
Matplotlib only.
