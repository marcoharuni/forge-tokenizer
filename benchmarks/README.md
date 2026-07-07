# Benchmarks

These scripts make local measurements on the current machine. They are not
production tokenizer benchmarks, and they are not claims about `tiktoken`,
HuggingFace Tokenizers, SentencePiece, or any other implementation.

Run:

```bash
python benchmarks/benchmark_encode.py
python benchmarks/benchmark_fertility.py
```

Both scripts train or use only the tiny local corpus included in this repository.
