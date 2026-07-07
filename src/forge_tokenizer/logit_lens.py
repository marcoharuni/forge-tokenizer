"""Simulated logit lens demonstration.

This module does not inspect a trained language model. It uses random matrices
and a gradually strengthened target direction to show the mechanics of projecting
residual stream states through an unembedding matrix.
"""

from __future__ import annotations

import numpy as np

from forge_tokenizer.unembedding import stable_softmax


def simulate_residual_stream(vocab_size: int = 12, d_model: int = 16, layers: int = 6, seed: int = 0):
    """Create simulated residual states, unembedding matrix, and labels."""

    rng = np.random.default_rng(seed)
    unembedding = rng.normal(size=(d_model, vocab_size))
    labels = [f"tok_{i}" for i in range(vocab_size)]
    target_id = vocab_size // 2
    target_direction = unembedding[:, target_id]
    target_direction = target_direction / (np.linalg.norm(target_direction) + 1e-12)
    states = []
    state = rng.normal(scale=0.4, size=d_model)
    for layer in range(layers):
        strength = (layer + 1) / layers
        state = 0.65 * state + strength * target_direction + rng.normal(scale=0.08, size=d_model)
        states.append(state.copy())
    return {"states": np.vstack(states), "unembedding": unembedding, "labels": labels, "target_id": target_id}


def run_simulated_logit_lens(vocab_size: int = 12, d_model: int = 16, layers: int = 6, seed: int = 0, top_k: int = 3):
    """Run the simulated logit lens and return top predictions by layer."""

    sim = simulate_residual_stream(vocab_size=vocab_size, d_model=d_model, layers=layers, seed=seed)
    rows = []
    for layer, state in enumerate(sim["states"]):
        logits = state @ sim["unembedding"]
        probs = stable_softmax(logits)
        order = np.argsort(-probs)[:top_k]
        rows.append(
            {
                "layer": layer,
                "predictions": [(sim["labels"][int(idx)], float(probs[idx])) for idx in order],
                "target": sim["labels"][sim["target_id"]],
            }
        )
    return rows


def format_layer_predictions(rows) -> str:
    """Format simulated layer predictions as readable text."""

    lines = ["Simulated logit lens: random residual stream, not a trained model."]
    for row in rows:
        preds = ", ".join(f"{token}={prob:.3f}" for token, prob in row["predictions"])
        lines.append(f"layer {row['layer']:02d}: {preds} | target={row['target']}")
    return "\n".join(lines)
