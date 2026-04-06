#!/usr/bin/env python3
"""Train a custom classifier on extracted embeddings.

Takes the embeddings and labels from extract_embeddings.py and trains
a multi-label classifier that can predict your custom genre tags.

Uses scikit-learn with a one-vs-rest approach — each tag gets its own
binary classifier, so a song can be predicted as multiple tags at once.

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --data data/ --min-samples 10
    python scripts/train_classifier.py --tags "Afro Trap,Arab Trap,Dark Rap,Neo Soul"
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

console = Console()

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"
TAXONOMY_PATH = Path(__file__).parent.parent / "taxonomy.yaml"


def load_retired_tags():
    """Load retired tag mapping from taxonomy."""
    with open(TAXONOMY_PATH) as f:
        taxonomy = yaml.safe_load(f)
    retired = {}
    for old, new in taxonomy.get("retired", {}).items():
        if new is None:
            continue
        if isinstance(new, list):
            retired[old] = new
        else:
            retired[old] = [new]
    return retired


def normalise_labels(labels: list[list[str]], retired: dict) -> list[list[str]]:
    """Clean up training labels — map retired tags, deduplicate."""
    cleaned = []
    for tag_list in labels:
        result = []
        for tag in tag_list:
            if tag in retired:
                for r in retired[tag]:
                    if r not in result:
                        result.append(r)
            else:
                if tag not in result:
                    result.append(tag)
        cleaned.append(result)
    return cleaned


def load_data(data_dir: Path):
    """Load embeddings and metadata."""
    embeddings_path = data_dir / "embeddings.npy"
    metadata_path = data_dir / "metadata.json"

    if not embeddings_path.exists() or not metadata_path.exists():
        console.print(f"[red]Data not found in {data_dir}/[/red]")
        console.print("Run extract_embeddings.py first.")
        sys.exit(1)

    embeddings = np.load(embeddings_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    return embeddings, metadata


def main():
    parser = argparse.ArgumentParser(description="Train custom genre classifier")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with embeddings.npy and metadata.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum samples per tag to include in training (default: 5)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated list of specific tags to train on (default: all)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data to hold out for testing (default: 0.2)",
    )
    args = parser.parse_args()

    # Load data
    console.print("Loading data...")
    embeddings, metadata = load_data(args.data)
    labels = metadata["labels"]  # list of lists

    console.print(f"  {embeddings.shape[0]} tracks, {embeddings.shape[1]}-dim embeddings")
    console.print(f"  {len(metadata['unique_tags'])} unique tags in dataset")

    # Normalise labels — map retired tags, deduplicate
    retired = load_retired_tags()
    labels = normalise_labels(labels, retired)
    console.print(f"  Labels normalised (retired tags mapped)")

    # Filter tags by minimum samples
    tag_counts = {}
    for tag_list in labels:
        for tag in tag_list:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    if args.tags:
        target_tags = [t.strip() for t in args.tags.split(",")]
        missing = [t for t in target_tags if t not in tag_counts]
        if missing:
            console.print(f"[yellow]Warning: tags not found in data: {missing}[/yellow]")
        target_tags = [t for t in target_tags if t in tag_counts]
    else:
        target_tags = [
            tag for tag, count in tag_counts.items() if count >= args.min_samples
        ]

    if not target_tags:
        console.print("[red]No tags meet the minimum sample threshold.[/red]")
        sys.exit(1)

    console.print(f"\nTraining on {len(target_tags)} tags (min {args.min_samples} samples each):")
    for tag in sorted(target_tags):
        console.print(f"  {tag}: {tag_counts.get(tag, 0)} samples")

    # Filter labels to only include target tags
    filtered_labels = []
    for tag_list in labels:
        filtered = [t for t in tag_list if t in target_tags]
        filtered_labels.append(filtered if filtered else ["_none_"])

    # Encode labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(filtered_labels)

    # Remove _none_ class if present
    if "_none_" in mlb.classes_:
        none_idx = list(mlb.classes_).index("_none_")
        y = np.delete(y, none_idx, axis=1)
        classes = [c for c in mlb.classes_ if c != "_none_"]
        mlb.classes_ = np.array(classes)

    console.print(f"\nLabel matrix: {y.shape} (tracks x tags)")

    X = embeddings

    # Train/test split
    n_test = int(len(X) * args.test_split)
    n_train = len(X) - n_test

    # Shuffle with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(X))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    console.print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Build classifier pipeline
    console.print("\nTraining MLP classifier...")
    classifier = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            OneVsRestClassifier(
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    learning_rate="adaptive",
                ),
                n_jobs=-1,
            ),
        ),
    ])

    classifier.fit(X_train, y_train)

    # Evaluate
    console.print("\n[bold]Evaluation on test set:[/bold]\n")
    y_pred = classifier.predict(X_test)

    # Per-tag results
    table = Table(title="Per-Tag Performance", show_header=True)
    table.add_column("Tag", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Support", justify="right", style="dim")

    report = classification_report(
        y_test, y_pred, target_names=mlb.classes_, output_dict=True, zero_division=0
    )

    for tag in sorted(mlb.classes_):
        if tag in report:
            r = report[tag]
            table.add_row(
                tag,
                f"{r['precision']:.2f}",
                f"{r['recall']:.2f}",
                f"{r['f1-score']:.2f}",
                str(int(r["support"])),
            )

    console.print(table)

    # Cross-validation on full dataset for more robust metrics
    console.print("\n[bold]Cross-validation (5-fold):[/bold]")
    for i, tag in enumerate(mlb.classes_):
        if y[:, i].sum() >= 10:  # Only CV if enough samples
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation="relu", max_iter=500,
                    random_state=42, early_stopping=True,
                    validation_fraction=0.1, learning_rate="adaptive",
                )),
            ])
            try:
                scores = cross_val_score(
                    pipe, X, y[:, i],
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring="f1",
                )
                console.print(f"  {tag:25s} F1: {scores.mean():.2f} (+/- {scores.std():.2f})")
            except Exception:
                console.print(f"  {tag:25s} [dim]skipped (insufficient class variety)[/dim]")

    # Save model
    args.output.mkdir(parents=True, exist_ok=True)
    model_data = {
        "classifier": classifier,
        "mlb": mlb,
        "tags": list(mlb.classes_),
        "embedding_dim": int(embeddings.shape[1]),
        "n_training_samples": len(X_train),
    }
    model_path = args.output / "custom_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    console.print(f"\n[green]Model saved to {model_path}[/green]")
    console.print(f"  Tags: {list(mlb.classes_)}")
    console.print(f"  Trained on {len(X_train)} tracks")


if __name__ == "__main__":
    main()
