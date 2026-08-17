"""Aggregate the multi-seed runs into mean +/- std tables and figures.

Seed 42 is reused from the original ``outputs/`` directory; the other seeds
are read from ``outputs_seed<seed>/`` directories produced by run_all_seeds.sh.
Results are written to ``outputs_multiseed/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SEED_DIRS = {
    42: Path("outputs"),
    123: Path("outputs_seed123"),
    2024: Path("outputs_seed2024"),
}
OUTPUT_DIR = Path("outputs_multiseed")


def load_table(filename: str) -> pd.DataFrame:
    frames = []
    for seed, directory in SEED_DIRS.items():
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run the full pipeline for seed {seed} first."
            )
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def mean_std(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    aggregations = {}
    for column in columns:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_std"] = (column, "std")
    result = frame.groupby(["fraction", "fraction_percent"], as_index=False).agg(
        **aggregations
    )
    return result.sort_values("fraction").reset_index(drop=True)


def plot_learning_curve(aggregate: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for column, label, marker in (
        ("accuracy", "Accuracy", "o"),
        ("macro_f1", "Macro-F1", "s"),
    ):
        ax.errorbar(
            aggregate["fraction_percent"],
            aggregate[f"{column}_mean"],
            yerr=aggregate[f"{column}_std"],
            marker=marker,
            capsize=4,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "learning_curve_meanstd.png", dpi=200)
    plt.close(fig)


def plot_shortcuts(aggregate: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for column, label, marker in (
        ("shortcut_rate", "Stopword + punctuation", "o"),
        ("stopword_rate", "Stopwords", "s"),
        ("punctuation_rate", "Punctuation", "^"),
    ):
        ax.errorbar(
            aggregate["fraction_percent"],
            aggregate[f"{column}_mean"] * 100,
            yerr=aggregate[f"{column}_std"] * 100,
            marker=marker,
            capsize=4,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Top-5 token rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "lime_shortcuts_meanstd.png", dpi=200)
    plt.close(fig)


def plot_deletion(aggregate: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        aggregate["fraction_percent"],
        aggregate["mean_probability_drop_mean"],
        yerr=aggregate["mean_probability_drop_std"],
        marker="o",
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Probability drop after deleting top-5 LIME tokens")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "deletion_drop_meanstd.png", dpi=200)
    plt.close(fig)


def pm(frame: pd.DataFrame, column: str) -> list[str]:
    return [
        f"{mean:.3f} ± {std:.3f}"
        for mean, std in zip(frame[f"{column}_mean"], frame[f"{column}_std"])
    ]


def write_markdown(
    metrics: pd.DataFrame,
    lime: pd.DataFrame,
    deletion: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Multi-seed aggregation (3 seeds: 42, 123, 2024)",
        "",
        "Seed 42 is the original run in `outputs/`; seeds 123 and 2024 are new",
        "runs with identical protocol and hyperparameters.",
        "",
        "| Training data (%) | Accuracy | Macro-F1 | Shortcut rate (LIME top-5) | Probability drop (deletion) | Flip rate (deletion) |",
        "|---|---|---|---|---|---|",
    ]
    for index, row in metrics.iterrows():
        lines.append(
            "| {pct:g}% | {acc} | {f1} | {shortcut} | {drop} | {flip} |".format(
                pct=row["fraction_percent"],
                acc=pm(metrics.loc[[index]], "accuracy")[0],
                f1=pm(metrics.loc[[index]], "macro_f1")[0],
                shortcut=pm(lime.loc[[index]], "shortcut_rate")[0],
                drop=pm(deletion.loc[[index]], "mean_probability_drop")[0],
                flip=pm(deletion.loc[[index]], "prediction_flip_rate")[0],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_table("metrics.csv")
    lime = load_table("lime_summary.csv")
    deletion = load_table("deletion_summary.csv")

    aggregate_metrics = mean_std(metrics, ["accuracy", "macro_f1", "precision", "recall"])
    aggregate_lime = mean_std(
        lime, ["shortcut_rate", "stopword_rate", "punctuation_rate"]
    )
    aggregate_deletion = mean_std(
        deletion, ["mean_probability_drop", "prediction_flip_rate"]
    )

    aggregate_metrics.to_csv(OUTPUT_DIR / "aggregate_metrics.csv", index=False)
    aggregate_lime.to_csv(OUTPUT_DIR / "aggregate_lime.csv", index=False)
    aggregate_deletion.to_csv(OUTPUT_DIR / "aggregate_deletion.csv", index=False)

    plot_learning_curve(aggregate_metrics, figures_dir)
    plot_shortcuts(aggregate_lime, figures_dir)
    plot_deletion(aggregate_deletion, figures_dir)
    write_markdown(
        aggregate_metrics,
        aggregate_lime,
        aggregate_deletion,
        OUTPUT_DIR / "aggregate_report.md",
    )

    print("\n=== Accuracy / Macro-F1 (mean ± std over 3 seeds) ===")
    print(
        aggregate_metrics[
            ["fraction_percent", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]
        ].to_string(index=False)
    )
    print("\n=== LIME shortcut rate (mean ± std over 3 seeds) ===")
    print(
        aggregate_lime[
            ["fraction_percent", "shortcut_rate_mean", "shortcut_rate_std"]
        ].to_string(index=False)
    )
    print("\n=== Deletion probability drop / flip rate (mean ± std over 3 seeds) ===")
    print(
        aggregate_deletion[
            [
                "fraction_percent",
                "mean_probability_drop_mean",
                "mean_probability_drop_std",
                "prediction_flip_rate_mean",
                "prediction_flip_rate_std",
            ]
        ].to_string(index=False)
    )
    print(f"\nAggregation saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
