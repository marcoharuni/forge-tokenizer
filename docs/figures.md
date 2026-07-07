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

The notebooks also generate inline figures:

- `04_token_tax_lab.ipynb` plots local token-tax multipliers.
- `05_token_id_to_meaning.ipynb` prints local embedding neighbors and sampling probabilities.
- `06_chat_formatting_token_tax.ipynb` plots token counts for plain text, code, JSON, and chat-style formatting.
