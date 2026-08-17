"""Minimal deletion test for the existing Text-CNN and LIME outputs.

For each saved model, this script removes the five LIME tokens from each of
the fixed explanation examples and measures the change in the probability of
the original predicted class. It does not retrain models or run new LIME
explanations.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from run_experiment import TOKEN_PATTERN, TextCNN, choose_device, predict_proba_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def delete_tokens(text: str, tokens: list[str]) -> str:
    """Remove all occurrences of the selected LIME token types."""
    selected = {token.strip().lower() for token in tokens}
    pieces: list[str] = []
    cursor = 0
    for match in TOKEN_PATTERN.finditer(text):
        if match.group().lower() in selected:
            pieces.append(text[cursor : match.start()])
            cursor = match.end()
    pieces.append(text[cursor:])
    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def load_model(
    checkpoint_path: Path,
    vocabulary_size: int,
    config: dict[str, object],
    device: torch.device,
) -> TextCNN:
    model = TextCNN(
        vocab_size=vocabulary_size,
        embedding_dim=int(config["embedding_dim"]),
        filters=int(config["filters"]),
        dropout=float(config["dropout"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    output_dir = args.output
    device = choose_device(args.device)
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((output_dir / "vocabulary.json").read_text(encoding="utf-8"))
    examples = pd.read_csv(output_dir / "lime_examples.csv")
    # keep_default_na=False: tokens like "nan" must stay strings, not become NaN.
    explanations = pd.read_csv(
        output_dir / "lime_explanations.csv", keep_default_na=False
    )

    max_length = int(config["max_length"])
    results: list[dict[str, object]] = []

    for fraction in sorted(explanations["fraction"].unique()):
        fraction_percent = fraction * 100
        checkpoint_path = output_dir / "models" / f"textcnn_{fraction_percent:g}pct.pt"
        model = load_model(checkpoint_path, len(vocabulary), config, device)
        current_explanations = explanations[explanations["fraction"] == fraction]
        print(f"Testing deletion for {fraction_percent:g}%")

        for example_number, (_, example) in enumerate(examples.iterrows(), start=1):
            row_id = int(example["row_id"])
            lime_rows = current_explanations[
                current_explanations["row_id"] == row_id
            ].sort_values("rank")
            top_tokens = lime_rows["token"].astype(str).tolist()[:5]
            deleted_text = delete_tokens(str(example["text"]), top_tokens)

            probabilities = predict_proba_texts(
                model,
                [str(example["text"]), deleted_text],
                vocabulary,
                max_length,
                device,
            )
            original_label = int(np.argmax(probabilities[0]))
            deleted_label = int(np.argmax(probabilities[1]))
            original_probability = float(probabilities[0, original_label])
            deleted_probability = float(probabilities[1, original_label])
            probability_drop = original_probability - deleted_probability

            results.append(
                {
                    "fraction": fraction,
                    "fraction_percent": fraction_percent,
                    "row_id": row_id,
                    "example_number": example_number,
                    "true_label": int(example["label"]),
                    "original_predicted_label": original_label,
                    "deleted_predicted_label": deleted_label,
                    "original_probability": original_probability,
                    "deleted_probability": deleted_probability,
                    "probability_drop": probability_drop,
                    "prediction_flipped": original_label != deleted_label,
                    "removed_tokens": " | ".join(top_tokens),
                    "deleted_text_length": len(deleted_text),
                }
            )
            if example_number % 25 == 0 or example_number == len(examples):
                print(f"  {example_number}/{len(examples)} examples")

    results_frame = pd.DataFrame(results).sort_values(["fraction", "row_id"])
    results_frame.to_csv(output_dir / "deletion_results.csv", index=False)
    summary = (
        results_frame.groupby(["fraction", "fraction_percent"], as_index=False)
        .agg(
            examples=("row_id", "count"),
            mean_probability_drop=("probability_drop", "mean"),
            median_probability_drop=("probability_drop", "median"),
            std_probability_drop=("probability_drop", "std"),
            positive_drop_rate=("probability_drop", lambda values: np.mean(values > 0)),
            prediction_flip_rate=("prediction_flipped", "mean"),
            mean_original_probability=("original_probability", "mean"),
            mean_deleted_probability=("deleted_probability", "mean"),
        )
        .sort_values("fraction")
    )
    summary.to_csv(output_dir / "deletion_summary.csv", index=False)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        summary["fraction_percent"],
        summary["mean_probability_drop"],
        yerr=summary["std_probability_drop"].fillna(0),
        marker="o",
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Probability drop after deleting top-5 LIME tokens")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "deletion_probability_drop.png", dpi=200)
    plt.close(fig)
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
