"""Matplotlib figures for examples and docs."""

from __future__ import annotations

from collections import Counter
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


def _finish_figure(fig, ax, path):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_vocab_params(output_path):
    """Plot parameter counts for several vocabulary sizes."""

    path = _prepare_path(output_path)
    vocab_sizes = np.array([1_000, 5_000, 10_000, 50_000])
    totals = [embedding_parameter_count(int(v), 256, tied=False)["total"] for v in vocab_sizes]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(vocab_sizes, totals, marker="o", linewidth=2.2, color="#2364aa")
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Parameters")
    ax.set_title("Embedding + unembedding parameters")
    return _finish_figure(fig, ax, path)


def plot_fertility_bar(results, output_path):
    """Plot fertility values by label."""

    path = _prepare_path(output_path)
    labels = list(results.keys())
    values = [results[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(labels, values, color="#2f855a", edgecolor="#1f2937", linewidth=0.8)
    ax.set_ylabel("Tokens per word")
    ax.set_title("Local fertility measurements")
    ax.tick_params(axis="x", rotation=30)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    return _finish_figure(fig, ax, path)


def plot_bpe_dropout_distribution(samples, output_path):
    """Plot a histogram of token counts from BPE dropout samples."""

    path = _prepare_path(output_path)
    counts = [len(sample) for sample in samples]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    if not counts:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return path

    frequencies = Counter(counts)
    xs = np.array(sorted(frequencies))
    ys = np.array([frequencies[x] for x in xs])
    bars = ax.bar(xs, ys, width=0.72, color="#c75146", edgecolor="#1f2937", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xlabel("Tokens produced for the same text")
    ax.set_ylabel("Number of seeded encodings")
    ax.set_title("BPE dropout changes segmentation length")
    for bar, value in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(int(value)), ha="center", va="bottom", fontsize=8)
    return _finish_figure(fig, ax, path)


def plot_embedding_neighbors_2d(embeddings, labels, output_path):
    """Plot the first two embedding dimensions with labels."""

    path = _prepare_path(output_path)
    emb = np.asarray(embeddings, dtype=float)
    if emb.shape[1] < 2:
        emb = np.pad(emb, ((0, 0), (0, 2 - emb.shape[1])))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(emb[:, 0], emb[:, 1], color="#2563eb", s=42, edgecolor="white", linewidth=0.8)
    offsets = [(7, 6), (7, -10), (-7, 6), (-7, -10), (0, 12), (0, -14), (13, 0), (-13, 0)]
    span_x = max(float(np.ptp(emb[:, 0])), 1e-9)
    span_y = max(float(np.ptp(emb[:, 1])), 1e-9)
    placed: list[tuple[float, float]] = []
    for x, y, label in zip(emb[:, 0], emb[:, 1], labels):
        nearby = sum(
            ((x - px) / span_x) ** 2 + ((y - py) / span_y) ** 2 < 0.012
            for px, py in placed
        )
        dx, dy = offsets[nearby % len(offsets)]
        ax.annotate(
            label,
            (x, y),
            fontsize=8,
            xytext=(dx, dy),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.78},
        )
        placed.append((float(x), float(y)))
    ax.set_title("PPMI/SVD embedding geometry")
    ax.set_xlabel("SVD dimension 1")
    ax.set_ylabel("SVD dimension 2")
    return _finish_figure(fig, ax, path)


def plot_temperature_distributions(logits, output_path):
    """Plot probability distributions for several temperatures."""

    path = _prepare_path(output_path)
    logits = np.asarray(logits, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for temperature in [0.5, 1.0, 2.0]:
        probs = logits_to_probs(logits, temperature=temperature)
        ax.plot(np.arange(len(probs)), probs, marker="o", linewidth=2, label=f"T={temperature}")
    ax.set_xlabel("Token index")
    ax.set_ylabel("Probability")
    ax.set_title("Temperature-scaled distributions")
    ax.legend(frameon=False)
    return _finish_figure(fig, ax, path)
