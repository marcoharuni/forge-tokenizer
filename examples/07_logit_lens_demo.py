from forge_tokenizer.logit_lens import format_layer_predictions, run_simulated_logit_lens
from forge_tokenizer.unembedding import logits_to_probs
from forge_tokenizer.visualization import plot_temperature_distributions


def main():
    rows = run_simulated_logit_lens(seed=7)
    print(format_layer_predictions(rows))
    fig = plot_temperature_distributions([0.2, 1.1, 2.4, -0.8, 0.0], "generated/temperature_distributions.png")
    probs = logits_to_probs([1000, 1001, 1002])
    print(f"stable_softmax_demo={[round(float(p), 4) for p in probs]}")
    print(f"figure={fig}")


if __name__ == "__main__":
    main()
