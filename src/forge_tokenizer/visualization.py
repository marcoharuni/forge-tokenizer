"""Matplotlib figures for examples and docs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "generated" / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

from forge_tokenizer.embeddings import embedding_parameter_count
from forge_tokenizer.unembedding import logits_to_probs


def _prepare_path(output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_vocab_params(output_path):
    """Plot parameter counts for several vocabulary sizes."""

    path = _prepare_path(output_path)
    vocab_sizes = np.array([1_000, 5_000, 10_000, 50_000])
    totals = [embedding_parameter_count(int(v), 256, tied=False)["total"] for v in vocab_sizes]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(vocab_sizes, totals, marker="o", color="#2a6f97")
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Parameters")
    ax.set_title("Embedding + unembedding parameters")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_fertility_bar(results, output_path):
    """Plot fertility values by label."""

    path = _prepare_path(output_path)
    labels = list(results.keys())
    values = [results[label] for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="#4c956c")
    ax.set_ylabel("Tokens per word")
    ax.set_title("Local fertility measurements")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_bpe_dropout_distribution(samples, output_path):
    """Plot a histogram of token counts from BPE dropout samples."""

    path = _prepare_path(output_path)
    counts = [len(sample) for sample in samples]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(counts, bins=range(min(counts), max(counts) + 2), color="#bc4749", alpha=0.85)
    ax.set_xlabel("Token count")
    ax.set_ylabel("Frequency")
    ax.set_title("BPE dropout token-count distribution")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_embedding_neighbors_2d(embeddings, labels, output_path):
    """Plot the first two embedding dimensions with labels."""

    path = _prepare_path(output_path)
    emb = np.asarray(embeddings, dtype=float)
    if emb.shape[1] < 2:
        emb = np.pad(emb, ((0, 0), (0, 2 - emb.shape[1])))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(emb[:, 0], emb[:, 1], color="#355070")
    for x, y, label in zip(emb[:, 0], emb[:, 1], labels):
        ax.annotate(label, (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_title("PPMI/SVD embedding geometry")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_temperature_distributions(logits, output_path):
    """Plot probability distributions for several temperatures."""

    path = _prepare_path(output_path)
    logits = np.asarray(logits, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    for temperature in [0.5, 1.0, 2.0]:
        probs = logits_to_probs(logits, temperature=temperature)
        ax.plot(np.arange(len(probs)), probs, marker="o", label=f"T={temperature}")
    ax.set_xlabel("Token index")
    ax.set_ylabel("Probability")
    ax.set_title("Temperature-scaled distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
