#!/usr/bin/env python3
"""Predict custom genre tags for untagged or new music.

Uses the trained classifier from train_classifier.py to suggest tags
with confidence scores. Supports dry-run preview and writing tags.

Usage:
    python scripts/predict_tags.py /path/to/music
    python scripts/predict_tags.py /path/to/music --threshold 0.6
    python scripts/predict_tags.py /path/to/music --apply
    python scripts/predict_tags.py /path/to/track.flac --verbose
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs
import mutagen
from mutagen.flac import FLAC
from mutagen.easyid3 import EasyID3

console = Console()

MODEL_DIR = Path(__file__).parent.parent / "models"
EFFNET_PATH = MODEL_DIR / "discogs-effnet-bs64-1.pb"
CLASSIFIER_PATH = MODEL_DIR / "custom_classifier.pkl"

AUDIO_EXTENSIONS = {".flac", ".mp3"}


def load_classifier():
    """Load the trained custom classifier."""
    if not CLASSIFIER_PATH.exists():
        console.print(f"[red]Trained model not found at {CLASSIFIER_PATH}[/red]")
        console.print("Run train_classifier.py first.")
        sys.exit(1)

    with open(CLASSIFIER_PATH, "rb") as f:
        model_data = pickle.load(f)

    return model_data


def get_existing_tags(filepath: Path) -> list[str]:
    """Read existing genre tags from file."""
    suffix = filepath.suffix.lower()
    try:
        if suffix == ".flac":
            audio = FLAC(str(filepath))
            if audio.tags is None:
                return []
            return [v for k, v in audio.tags if k.upper() == "GENRE"]
        elif suffix == ".mp3":
            audio = EasyID3(str(filepath))
            return list(audio.get("genre", []))
    except Exception:
        pass
    return []


def extract_embedding(filepath: Path) -> np.ndarray | None:
    """Extract embedding from a single track."""
    try:
        audio = MonoLoader(
            filename=str(filepath), sampleRate=16000, resampleQuality=4
        )()
        embeddings = TensorflowPredictEffnetDiscogs(
            graphFilename=str(EFFNET_PATH),
            output="PartitionedCall:1",
        )(audio)
        return np.mean(embeddings, axis=0)
    except Exception as e:
        console.print(f"[red]  Error: {e}[/red]")
        return None


def write_tags(filepath: Path, new_tags: list[str], merge: bool = True):
    """Write genre tags to a music file."""
    suffix = filepath.suffix.lower()

    if suffix == ".flac":
        audio = FLAC(str(filepath))
        if audio.tags is None:
            audio.add_tags()

        if merge:
            existing = [v for k, v in audio.tags if k.upper() == "GENRE"]
            all_tags = list(dict.fromkeys(existing + new_tags))  # dedupe, preserve order
        else:
            all_tags = new_tags

        # Remove existing GENRE entries
        audio.tags._data = [
            (k, v) for k, v in audio.tags._data if k.upper() != "GENRE"
        ]
        # Write new ones
        for tag in all_tags:
            audio.tags.append(("GENRE", tag))
        audio.save()

    elif suffix == ".mp3":
        audio = EasyID3(str(filepath))
        if merge:
            existing = list(audio.get("genre", []))
            all_tags = list(dict.fromkeys(existing + new_tags))
        else:
            all_tags = new_tags
        audio["genre"] = all_tags
        audio.save()


def find_audio_files(path: Path) -> list[Path]:
    """Find audio files."""
    if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
        return [path]
    if path.is_dir():
        files = []
        for ext in AUDIO_EXTENSIONS:
            files.extend(path.rglob(f"*{ext}"))
        return sorted(files)
    return []


def main():
    parser = argparse.ArgumentParser(description="Predict custom genre tags")
    parser.add_argument("path", type=Path, help="File or folder to predict tags for")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum confidence to suggest a tag (default: 0.5)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write predicted tags to files (default: dry-run preview)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing genre tags instead of merging",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all tag confidences, not just above threshold",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save predictions to JSON file",
    )
    args = parser.parse_args()

    if not EFFNET_PATH.exists():
        console.print(f"[red]Essentia model not found at {EFFNET_PATH}[/red]")
        sys.exit(1)

    # Load classifier
    model_data = load_classifier()
    classifier = model_data["classifier"]
    tags = model_data["tags"]
    console.print(f"Loaded classifier with {len(tags)} tags: {tags}")

    # Find files
    files = find_audio_files(args.path)
    if not files:
        console.print(f"[red]No audio files found at {args.path}[/red]")
        sys.exit(1)

    console.print(f"Processing {len(files)} file(s)...\n")

    # Process
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Predicting", total=len(files))

        for filepath in files:
            progress.update(task, description=f"[cyan]{filepath.name[:40]}[/cyan]")

            existing = get_existing_tags(filepath)
            embedding = extract_embedding(filepath)

            if embedding is None:
                progress.advance(task)
                continue

            # Get probability estimates
            embedding_2d = embedding.reshape(1, -1)

            # Try to get probability estimates
            try:
                probas = classifier.predict_proba(embedding_2d)[0]
            except AttributeError:
                # Fallback: use decision function
                decision = classifier.decision_function(embedding_2d)[0]
                # Sigmoid to convert to probabilities
                probas = 1 / (1 + np.exp(-decision))

            # Build predictions
            predictions = []
            for i, tag in enumerate(tags):
                conf = float(probas[i]) if i < len(probas) else 0.0
                predictions.append({"tag": tag, "confidence": conf})

            predictions.sort(key=lambda x: -x["confidence"])

            suggested = [p for p in predictions if p["confidence"] >= args.threshold]
            suggested_tags = [p["tag"] for p in suggested]

            # New tags = suggested minus already existing
            new_tags = [t for t in suggested_tags if t not in existing]

            result = {
                "file": str(filepath),
                "filename": filepath.name,
                "existing_tags": existing,
                "predictions": predictions,
                "suggested": suggested,
                "new_tags": new_tags,
            }
            results.append(result)
            progress.advance(task)

    # Display results
    console.print()

    for result in results:
        table = Table(title=result["filename"], show_header=True)
        table.add_column("Tag", style="cyan")
        table.add_column("Confidence", justify="right")
        table.add_column("Status", style="dim")

        display = result["predictions"] if args.verbose else result["suggested"]

        for pred in display:
            tag = pred["tag"]
            conf = f"{pred['confidence']:.1%}"

            if tag in result["existing_tags"]:
                status = "already tagged"
                style = "dim"
            elif pred["confidence"] >= args.threshold:
                status = "NEW"
                style = "green bold"
            else:
                status = ""
                style = "dim"

            table.add_row(f"[{style}]{tag}[/{style}]", conf, status)

        if result["existing_tags"]:
            table.add_row(
                "[dim]Existing[/dim]",
                "",
                ", ".join(result["existing_tags"]),
            )

        console.print(table)
        console.print()

    # Summary
    tracks_with_suggestions = [r for r in results if r["new_tags"]]
    total_new = sum(len(r["new_tags"]) for r in results)

    console.print(f"[bold]Summary:[/bold]")
    console.print(f"  Processed: {len(results)} tracks")
    console.print(f"  Tracks with new suggestions: {len(tracks_with_suggestions)}")
    console.print(f"  Total new tags to add: {total_new}")

    # Save JSON output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"  Predictions saved to {args.output}")

    # Apply tags
    if args.apply and total_new > 0:
        console.print()
        if not Confirm.ask(f"Write {total_new} new tags to {len(tracks_with_suggestions)} files?"):
            console.print("[yellow]Aborted.[/yellow]")
            return

        written = 0
        for result in tracks_with_suggestions:
            filepath = Path(result["file"])
            try:
                write_tags(filepath, result["new_tags"], merge=not args.replace)
                written += 1
                console.print(
                    f"  [green]✓[/green] {result['filename']}: "
                    f"+{', '.join(result['new_tags'])}"
                )
            except Exception as e:
                console.print(f"  [red]✗[/red] {result['filename']}: {e}")

        console.print(f"\n[green]Written tags to {written} files.[/green]")

    elif not args.apply and total_new > 0:
        console.print(
            "\n[yellow]Dry run — use --apply to write tags to files.[/yellow]"
        )


if __name__ == "__main__":
    main()
