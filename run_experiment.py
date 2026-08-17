"""Reproducible data-scaling experiment for a Text-CNN on IMDB reviews.

The script keeps the tokenizer, model architecture, validation split, test split,
and LIME examples fixed while changing only the amount of supervised training
data. Results are written to an output directory for later use in the report.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("IMDB Dataset.csv"),
        help="Path to the IMDB CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory where models, tables, and figures are saved.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--filters", type=int, default=100)
    parser.add_argument("--max-vocab", type=int, default=20_000)
    parser.add_argument("--max-length", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lime-examples", type=int, default=100)
    parser.add_argument("--lime-samples", type=int, default=1000)
    parser.add_argument("--skip-lime", action="store_true")
    parser.add_argument(
        "--lime-only",
        action="store_true",
        help="Reuse checkpoints in --output and run only the LIME analysis.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not hasattr(torch, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clean_review(text: str) -> str:
    """Remove markup while preserving ordinary words and punctuation."""
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def lime_tokenize(text: str) -> list[str]:
    """Return exact word and punctuation spans for LIME's callable tokenizer."""
    return TOKEN_PATTERN.findall(text)


def load_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    expected = {"review", "sentiment"}
    missing = expected.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data[["review", "sentiment"]].copy()
    data["row_id"] = np.arange(len(data), dtype=np.int64)
    data["text"] = data["review"].map(clean_review)
    data["label"] = data["sentiment"].map({"negative": 0, "positive": 1})
    if data["label"].isna().any():
        raise ValueError("The sentiment column contains labels other than positive/negative.")
    data["label"] = data["label"].astype(np.int64)
    return data


def make_splits(data: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_pool, test = train_test_split(
        data,
        test_size=0.10,
        stratify=data["label"],
        random_state=seed,
    )
    train, validation = train_test_split(
        train_pool,
        test_size=1 / 9,
        stratify=train_pool["label"],
        random_state=seed,
    )
    return (
        train.sort_values("row_id").reset_index(drop=True),
        validation.sort_values("row_id").reset_index(drop=True),
        test.sort_values("row_id").reset_index(drop=True),
    )


def make_nested_subsets(train: pd.DataFrame, seed: int) -> dict[float, pd.DataFrame]:
    shuffled_by_label = {
        label: train[train["label"] == label]
        .sample(frac=1.0, random_state=seed + label + 1)
        .reset_index(drop=True)
        for label in (0, 1)
    }
    subsets: dict[float, pd.DataFrame] = {}
    for fraction in FRACTIONS:
        target_size = int(round(len(train) * fraction))
        target_size = max(2, target_size - (target_size % 2))
        per_class = target_size // 2
        selected = pd.concat(
            [shuffled_by_label[label].iloc[:per_class] for label in (0, 1)],
            ignore_index=True,
        )
        subsets[fraction] = selected.sort_values("row_id").reset_index(drop=True)
    return subsets


def build_vocabulary(texts: Iterable[str], max_vocab: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    ordered_tokens = sorted(counts, key=lambda token: (-counts[token], token))
    for token in ordered_tokens[: max(0, max_vocab - len(vocabulary))]:
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(text: str, vocabulary: dict[str, int], max_length: int) -> torch.Tensor:
    ids = [vocabulary.get(token, vocabulary["<UNK>"]) for token in tokenize(text)]
    ids = ids[:max_length]
    ids.extend([vocabulary["<PAD>"]] * (max_length - len(ids)))
    return torch.tensor(ids, dtype=torch.long)


class ReviewDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        vocabulary: dict[str, int],
        max_length: int,
    ) -> None:
        self.texts = frame["text"].tolist()
        self.labels = frame["label"].astype(np.int64).tolist()
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            encode_text(self.texts[index], self.vocabulary, self.max_length),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        filters: int,
        kernels: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, filters, kernel_size=kernel)
            for kernel in kernels
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(filters * len(kernels), 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = [torch.relu(conv(embedded)).amax(dim=2) for conv in self.convolutions]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


def create_loader(
    frame: pd.DataFrame,
    vocabulary: dict[str, int],
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = ReviewDataset(frame, vocabulary, max_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    labels: list[int] = []
    with torch.inference_mode():
        for tokens, batch_labels in loader:
            tokens = tokens.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(tokens)
            losses.append(float(criterion(logits, batch_labels).item()) * len(batch_labels))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
    return {
        "loss": sum(losses) / len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }


def train_model(
    model: TextCNN,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> tuple[TextCNN, dict[str, float | int]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        for tokens, labels in train_loader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens), labels)
            loss.backward()
            optimizer.step()

        validation_metrics = evaluate_model(model, validation_loader, device, criterion)
        history.append({"epoch": epoch, **validation_metrics})
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_loss": best_validation_loss,
        "history": history,
    }


def predict_proba_texts(
    model: nn.Module,
    texts: list[str],
    vocabulary: dict[str, int],
    max_length: int,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    tokens = torch.stack([encode_text(text, vocabulary, max_length) for text in texts])
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            batch = tokens[start : start + batch_size].to(device)
            probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def select_lime_examples(test: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count % 2:
        raise ValueError("--lime-examples must be even for a balanced sample.")
    per_class = count // 2
    selected = pd.concat(
        [
            test[test["label"] == label]
            .sample(n=per_class, random_state=seed + label + 100)
            for label in (0, 1)
        ],
        ignore_index=True,
    )
    return selected.sort_values(["label", "row_id"]).reset_index(drop=True)


def token_flags(token: str) -> tuple[bool, bool]:
    normalized = token.strip().lower()
    is_stopword = normalized in ENGLISH_STOP_WORDS
    is_punctuation = bool(normalized) and all(not character.isalnum() for character in normalized)
    return is_stopword, is_punctuation


def run_lime(
    model: nn.Module,
    examples: pd.DataFrame,
    vocabulary: dict[str, int],
    max_length: int,
    device: torch.device,
    fraction: float,
    seed: int,
    num_samples: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError as error:
        raise RuntimeError(
            "LIME is not installed. Run `python3 -m pip install -r requirements.txt`."
        ) from error

    explainer = LimeTextExplainer(
        class_names=["negative", "positive"],
        split_expression=lime_tokenize,
        bow=True,
        random_state=seed,
    )
    explanation_rows: list[dict[str, object]] = []
    local_scores: list[float] = []
    total_tokens = 0
    stopword_tokens = 0
    punctuation_tokens = 0

    for example_number, (_, example) in enumerate(examples.iterrows(), start=1):
        text = str(example["text"])
        probabilities = predict_proba_texts(
            model, [text], vocabulary, max_length, device
        )[0]
        predicted_label = int(np.argmax(probabilities))
        explanation = explainer.explain_instance(
            text,
            lambda batch: predict_proba_texts(
                model, list(batch), vocabulary, max_length, device
            ),
            labels=(predicted_label,),
            num_features=5,
            num_samples=num_samples,
        )
        terms = sorted(
            explanation.as_list(label=predicted_label),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5]
        local_score = float(explanation.score)
        local_scores.append(local_score)
        for rank, (token, weight) in enumerate(terms, start=1):
            is_stopword, is_punctuation = token_flags(token)
            total_tokens += 1
            stopword_tokens += int(is_stopword)
            punctuation_tokens += int(is_punctuation)
            explanation_rows.append(
                {
                    "fraction": fraction,
                    "fraction_percent": fraction * 100,
                    "row_id": int(example["row_id"]),
                    "example_number": example_number,
                    "true_label": int(example["label"]),
                    "predicted_label": predicted_label,
                    "prediction_confidence": float(probabilities[predicted_label]),
                    "rank": rank,
                    "token": token,
                    "weight": float(weight),
                    "is_stopword": is_stopword,
                    "is_punctuation": is_punctuation,
                    "is_surface_shortcut": is_stopword or is_punctuation,
                    "local_surrogate_r2": local_score,
                }
            )
        if example_number % 10 == 0 or example_number == len(examples):
            print(
                f"  LIME {fraction * 100:g}%: {example_number}/{len(examples)} examples"
            )

    summary = {
        "fraction": fraction,
        "fraction_percent": fraction * 100,
        "examples": len(examples),
        "top_tokens": total_tokens,
        "stopword_rate": stopword_tokens / total_tokens if total_tokens else 0.0,
        "punctuation_rate": punctuation_tokens / total_tokens if total_tokens else 0.0,
        "shortcut_rate": (stopword_tokens + punctuation_tokens) / total_tokens
        if total_tokens
        else 0.0,
        "non_shortcut_rate": 1
        - ((stopword_tokens + punctuation_tokens) / total_tokens)
        if total_tokens
        else 0.0,
        "mean_local_surrogate_r2": float(np.mean(local_scores)) if local_scores else 0.0,
    }
    return explanation_rows, summary


def save_json(data: object, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_results(metrics: pd.DataFrame, lime_summary: pd.DataFrame | None, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = metrics.sort_values("fraction")
    x = metrics["fraction_percent"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, metrics["accuracy"], marker="o", label="Accuracy")
    ax.plot(x, metrics["macro_f1"], marker="s", label="Macro-F1")
    ax.set_xscale("log")
    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "learning_curve.png", dpi=200)
    plt.close(fig)

    if lime_summary is None or lime_summary.empty:
        return

    lime_summary = lime_summary.sort_values("fraction")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        lime_summary["fraction_percent"],
        lime_summary["shortcut_rate"] * 100,
        marker="o",
        label="Stopword + punctuation",
    )
    ax.plot(
        lime_summary["fraction_percent"],
        lime_summary["stopword_rate"] * 100,
        marker="s",
        label="Stopwords",
    )
    ax.plot(
        lime_summary["fraction_percent"],
        lime_summary["punctuation_rate"] * 100,
        marker="^",
        label="Punctuation",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Top-5 token rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "lime_shortcuts_curve.png", dpi=200)
    plt.close(fig)

    merged = metrics.merge(
        lime_summary[["fraction", "shortcut_rate"]], on="fraction", how="inner"
    )
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(
        merged["fraction_percent"],
        merged["accuracy"],
        marker="o",
        color="tab:blue",
        label="Accuracy",
    )
    ax1.set_xlabel("Training data (%)")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(
        merged["fraction_percent"],
        merged["shortcut_rate"] * 100,
        marker="s",
        color="tab:red",
        label="Shortcut rate",
    )
    ax2.set_ylabel("Shortcut rate (%)", color="tab:red")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor="tab:red")
    fig.tight_layout()
    fig.savefig(figures_dir / "accuracy_vs_shortcuts.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.lime_only and args.skip_lime:
        raise ValueError("--lime-only and --skip-lime cannot be used together.")
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = args.output
    models_dir = output_dir / "models"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["data"] = str(args.data)
    config["output"] = str(args.output)
    config["device_used"] = str(device)
    config["fractions"] = list(FRACTIONS)
    save_json(config, output_dir / "config.json")

    print(f"Device: {device}")
    print(f"Loading dataset: {args.data}")
    data = load_dataset(args.data)
    train, validation, test = make_splits(data, args.seed)
    subsets = make_nested_subsets(train, args.seed)
    vocabulary = build_vocabulary(train["text"], args.max_vocab)
    save_json(vocabulary, output_dir / "vocabulary.json")

    split_frames = []
    for split_name, frame in (("train", train), ("validation", validation), ("test", test)):
        split = frame[["row_id", "label"]].copy()
        split["split"] = split_name
        split_frames.append(split)
    pd.concat(split_frames, ignore_index=True).to_csv(output_dir / "splits.csv", index=False)

    lime_examples = None
    if not args.skip_lime:
        lime_examples = select_lime_examples(test, args.lime_examples, args.seed)
        lime_examples[["row_id", "label", "text"]].to_csv(
            output_dir / "lime_examples.csv", index=False
        )

    validation_loader = create_loader(
        validation,
        vocabulary,
        args.max_length,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    if args.lime_only:
        metrics_path = output_dir / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Cannot run --lime-only: missing existing metrics file {metrics_path}."
            )
        metrics_rows = pd.read_csv(metrics_path).to_dict("records")
    else:
        metrics_rows = []
    lime_rows: list[dict[str, object]] = []
    lime_summary_rows: list[dict[str, object]] = []

    for fraction, subset in subsets.items():
        set_seed(args.seed)
        fraction_percent = fraction * 100
        print(f"\nTraining fraction: {fraction_percent:g}% ({len(subset)} examples)")
        test_loader = create_loader(
            test,
            vocabulary,
            args.max_length,
            args.batch_size,
            shuffle=False,
            seed=args.seed,
        )
        model = TextCNN(
            vocab_size=len(vocabulary),
            embedding_dim=args.embedding_dim,
            filters=args.filters,
            dropout=args.dropout,
        ).to(device)
        checkpoint_path = models_dir / f"textcnn_{fraction_percent:g}pct.pt"
        if args.lime_only:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            print(f"  Loaded checkpoint: {checkpoint_path.name}")
        else:
            train_loader = create_loader(
                subset,
                vocabulary,
                args.max_length,
                args.batch_size,
                shuffle=True,
                seed=args.seed,
            )
            start_time = time.perf_counter()
            model, training_info = train_model(
                model,
                train_loader,
                validation_loader,
                device,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                patience=args.patience,
            )
            training_seconds = time.perf_counter() - start_time
            test_metrics = evaluate_model(model, test_loader, device, nn.CrossEntropyLoss())
            metrics_rows.append(
                {
                    "fraction": fraction,
                    "fraction_percent": fraction_percent,
                    "train_size": len(subset),
                    "seed": args.seed,
                    "training_seconds": training_seconds,
                    **{key: value for key, value in training_info.items() if key != "history"},
                    "test_loss": test_metrics["loss"],
                    "accuracy": test_metrics["accuracy"],
                    "macro_f1": test_metrics["macro_f1"],
                    "precision": test_metrics["precision"],
                    "recall": test_metrics["recall"],
                }
            )
            save_json(
                training_info["history"],
                output_dir / f"history_{fraction_percent:g}pct.json",
            )
            torch.save(
                {
                    "model_state": {
                        name: parameter.detach().cpu()
                        for name, parameter in model.state_dict().items()
                    },
                    "fraction": fraction,
                    "vocab_size": len(vocabulary),
                    "args": config,
                },
                checkpoint_path,
            )
            print(
                f"  test accuracy={test_metrics['accuracy']:.4f}, "
                f"macro-F1={test_metrics['macro_f1']:.4f}, "
                f"time={training_seconds:.1f}s"
            )

        if lime_examples is not None:
            current_lime_rows, current_summary = run_lime(
                model,
                lime_examples,
                vocabulary,
                args.max_length,
                device,
                fraction,
                args.seed,
                args.lime_samples,
            )
            lime_rows.extend(current_lime_rows)
            lime_summary_rows.append(current_summary)

    metrics = pd.DataFrame(metrics_rows).sort_values("fraction")
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    lime_summary = None
    if lime_rows:
        pd.DataFrame(lime_rows).to_csv(output_dir / "lime_explanations.csv", index=False)
        lime_summary = pd.DataFrame(lime_summary_rows).sort_values("fraction")
        lime_summary.to_csv(output_dir / "lime_summary.csv", index=False)
    plot_results(metrics, lime_summary, figures_dir)
    print(f"\nResults saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
