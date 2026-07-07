# Experiments

## BPE merge trace

`examples/01_train_toy_bpe.py` trains byte-level BPE on `data/tiny_corpus.txt`
and prints the first learned merges. The trace is deterministic: the most
frequent pair wins, and ties are resolved lexicographically by bytes.

`notebooks/03_bpe_merge_trace.ipynb` slows this down further. It shows the
pre-tokens, the initial UTF-8 bytes, adjacent pair counts, the selected merge,
and the segmentation after several merge steps.

## Fertility

`examples/03_measure_fertility.py` computes tokens per word for local
multilingual samples. These values depend on the tiny training corpus and should
not be generalized.

`notebooks/04_token_tax_lab.ipynb` adds character counts, word counts,
compression ratio, single-token retention rate, and a local multiplier versus
English.

## Token tax

The fertility and formatting-overhead metrics show how different scripts,
languages, and code formatting can require different token budgets.

`notebooks/06_chat_formatting_token_tax.ipynb` compares plain text, code, JSON,
and chat-style message formatting with the same local tokenizer.

## BPE dropout

`examples/04_bpe_dropout.py` encodes the same text with seeded BPE dropout.
Skipping merge rules can produce longer or different segmentations.

## PPMI/SVD embedding analogies

`examples/05_ppmi_svd_embeddings.py` and `examples/06_embedding_neighbors.py`
build co-occurrence counts, transform them with PPMI, and use SVD to create
small embeddings from the included corpus.

`notebooks/05_token_id_to_meaning.ipynb` connects token IDs to one-hot vectors,
embedding lookup, cosine neighbors, simulated unembedding logits, softmax, and
sampling.

## Stable softmax

`forge_tokenizer.unembedding.stable_softmax` subtracts the maximum logit before
exponentiation so large logits such as `[1000, 1001, 1002]` remain finite.

## Simulated logit lens

`examples/07_logit_lens_demo.py` uses random matrices and a target direction to
demonstrate the projection mechanism. It is explicitly not a real trained model.
