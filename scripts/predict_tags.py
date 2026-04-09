#!/usr/bin/env python3
"""Predict custom genre tags for untagged or new music.

Uses the trained classifier from train_classifier.py to suggest tags
with confidence scores. Two-tier system: confident tags auto-apply,
borderline tags are shown for review.

Usage:
    python scripts/predict_tags.py /path/to/music --apply
    python scripts/predict_tags.py /path/to/music --apply --auto-only
    python scripts/predict_tags.py /path/to/music --threshold 0.15 --apply
    python scripts/predict_tags.py /path/to/track.flac --verbose
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
TAXONOMY_PATH = Path(__file__).parent.parent / "taxonomy.yaml"

AUDIO_EXTENSIONS = {".flac", ".mp3"}


def load_taxonomy():
    """Load taxonomy for tag validation and retired tag mapping."""
    with open(TAXONOMY_PATH) as f:
        taxonomy = yaml.safe_load(f)

    # Build set of all valid tags
    valid_tags = set()
    valid_tags.update(taxonomy.get("base_genres", []))
    for sg_list in taxonomy.get("subgenres", {}).values():
        if isinstance(sg_list, list):
            valid_tags.update(sg_list)
    valid_tags.update(taxonomy.get("regional", []))
    valid_tags.update(taxonomy.get("scene", []))
    valid_tags.update(taxonomy.get("style", []))
    valid_tags.update(taxonomy.get("mood", []))

    # Build retired tag mapping
    retired = {}
    for old, new in taxonomy.get("retired", {}).items():
        if new is None:
            continue  # tag retired with no replacement
        if isinstance(new, list):
            retired[old] = new  # maps to multiple tags
        else:
            retired[old] = [new]  # maps to single tag

    return valid_tags, retired


def normalise_tags(tags: list[str], valid_tags: set, retired: dict) -> list[str]:
    """Validate and normalise tags against the taxonomy.

    - Maps retired tags to their replacements
    - Warns about unknown tags
    - Deduplicates while preserving order
    """
    result = []
    for tag in tags:
        if tag in retired:
            replacements = retired[tag]
            for r in replacements:
                if r not in result:
                    result.append(r)
        elif tag in valid_tags:
            if tag not in result:
                result.append(tag)
        else:
            # Unknown tag — still include but warn
            console.print(f"[yellow]  Warning: '{tag}' not in taxonomy[/yellow]")
            if tag not in result:
                result.append(tag)
    return result


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


def write_tags(filepath: Path, new_tags: list[str], merge: bool = True,
               valid_tags: set = None, retired: dict = None):
    """Write genre tags to a music file.

    Handles:
    - Retired tag mapping (e.g. 'Hip-Hop' → 'Hip-Hop/Rap')
    - Deduplication
    - Mixed-case GENRE key normalisation (lowercase 'genre' → uppercase 'GENRE')
    - Proper multi-value FLAC Vorbis entries
    """
    suffix = filepath.suffix.lower()

    # Normalise new tags against taxonomy
    if valid_tags and retired:
        new_tags = normalise_tags(new_tags, valid_tags, retired)

    if suffix == ".flac":
        audio = FLAC(str(filepath))
        if audio.tags is None:
            audio.add_tags()

        if merge:
            # Read existing, matching both 'genre' and 'GENRE'
            existing = [v for k, v in audio.tags if k.upper() == "GENRE"]
            # Normalise existing tags too (fix any retired tags already on file)
            if valid_tags and retired:
                existing = normalise_tags(existing, valid_tags, retired)
            all_tags = list(dict.fromkeys(existing + new_tags))  # dedupe, preserve order
        else:
            all_tags = new_tags

        # Remove ALL genre entries (both 'genre' and 'GENRE' — normalise to uppercase)
        audio.tags._data = [
            (k, v) for k, v in audio.tags._data if k.upper() != "GENRE"
        ]
        # Write as uppercase GENRE entries
        for tag in all_tags:
            audio.tags.append(("GENRE", tag))
        audio.save()

    elif suffix == ".mp3":
        audio = EasyID3(str(filepath))
        if merge:
            existing = list(audio.get("genre", []))
            if valid_tags and retired:
                existing = normalise_tags(existing, valid_tags, retired)
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
    parser = argparse.ArgumentParser(
        description="Predict custom genre tags",
        epilog="Two-tier mode (default): tags above --auto are written automatically, "
               "tags between --review and --auto are prompted for confirmation. "
               "Use --threshold to bypass two-tier mode with a single hard cutoff.",
    )
    parser.add_argument("path", type=Path, help="File or folder to predict tags for")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Hard cutoff — bypass two-tier mode, treat everything above this as NEW",
    )
    parser.add_argument(
        "--auto",
        type=float,
        default=0.5,
        help="Auto-apply threshold — tags above this are written without prompting (default: 0.5)",
    )
    parser.add_argument(
        "--review",
        type=float,
        default=0.15,
        help="Review threshold — tags between this and --auto are prompted (default: 0.15)",
    )
    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="Skip the review tier — only apply confident tags above --auto",
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

    # Determine mode
    if args.threshold is not None:
        # Single threshold bypass mode
        auto_threshold = args.threshold
        review_threshold = args.threshold  # same — no review tier
        two_tier = False
    else:
        auto_threshold = args.auto
        review_threshold = args.review
        two_tier = not args.auto_only

    if not EFFNET_PATH.exists():
        console.print(f"[red]Essentia model not found at {EFFNET_PATH}[/red]")
        sys.exit(1)

    # Load taxonomy for tag validation
    valid_tags, retired = load_taxonomy()

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

            # Two-tier: split into confident and review
            confident = [p for p in predictions if p["confidence"] >= auto_threshold]
            review = [p for p in predictions
                      if review_threshold <= p["confidence"] < auto_threshold] if two_tier else []

            confident_tags = [p["tag"] for p in confident if p["tag"] not in existing]
            review_tags = [p["tag"] for p in review if p["tag"] not in existing]

            result = {
                "file": str(filepath),
                "filename": filepath.name,
                "existing_tags": existing,
                "predictions": predictions,
                "confident": confident,
                "review": review,
                "confident_tags": confident_tags,
                "review_tags": review_tags,
                "new_tags": confident_tags,  # start with confident only
            }
            results.append(result)
            progress.advance(task)

    # Display results
    console.print()

    for result in results:
        # Skip tracks with nothing to show (unless verbose)
        if not args.verbose and not result["confident_tags"] and not result["review_tags"]:
            continue

        table = Table(title=result["filename"], show_header=True)
        table.add_column("Tag", style="cyan")
        table.add_column("Confidence", justify="right")
        table.add_column("Status", style="dim")

        if args.verbose:
            display = result["predictions"]
        else:
            display = result["confident"] + result["review"]

        for pred in display:
            tag = pred["tag"]
            conf = f"{pred['confidence']:.1%}"

            if tag in result["existing_tags"]:
                status = "already tagged"
                style = "dim"
            elif pred["confidence"] >= auto_threshold:
                status = "AUTO"
                style = "green bold"
            elif two_tier and pred["confidence"] >= review_threshold:
                status = "REVIEW"
                style = "yellow"
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
    total_confident = sum(len(r["confident_tags"]) for r in results)
    total_review = sum(len(r["review_tags"]) for r in results)
    tracks_with_confident = [r for r in results if r["confident_tags"]]
    tracks_with_review = [r for r in results if r["review_tags"]]

    console.print(f"[bold]Summary:[/bold]")
    console.print(f"  Processed: {len(results)} tracks")
    if two_tier:
        console.print(f"  [green]AUTO[/green]:   {total_confident} tags on {len(tracks_with_confident)} tracks (above {auto_threshold:.0%})")
        console.print(f"  [yellow]REVIEW[/yellow]: {total_review} tags on {len(tracks_with_review)} tracks ({review_threshold:.0%}–{auto_threshold:.0%})")
    else:
        console.print(f"  New tags: {total_confident} on {len(tracks_with_confident)} tracks (above {auto_threshold:.0%})")

    # Save JSON output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"  Predictions saved to {args.output}")

    if not args.apply:
        if total_confident + total_review > 0:
            console.print(
                "\n[yellow]Dry run — use --apply to write tags to files.[/yellow]"
            )
        return

    # Apply tags
    written = 0

    # 1. Auto-apply confident tags
    if total_confident > 0:
        console.print(f"\n[green bold]Writing {total_confident} confident tags...[/green bold]")
        for result in tracks_with_confident:
            filepath = Path(result["file"])
            try:
                write_tags(filepath, result["confident_tags"], merge=not args.replace,
                           valid_tags=valid_tags, retired=retired)
                written += 1
                console.print(
                    f"  [green]✓[/green] {result['filename']}: "
                    f"+{', '.join(result['confident_tags'])}"
                )
            except Exception as e:
                console.print(f"  [red]✗[/red] {result['filename']}: {e}")

    # 2. Review tier — prompt per track
    if two_tier and total_review > 0:
        console.print(f"\n[yellow bold]Review {total_review} borderline tags:[/yellow bold]\n")

        for result in tracks_with_review:
            filepath = Path(result["file"])
            review_preds = [p for p in result["review"]
                           if p["tag"] in result["review_tags"]]

            console.print(f"  [cyan]{result['filename']}[/cyan]")
            for pred in review_preds:
                console.print(f"    {pred['tag']:25s} {pred['confidence']:.1%}")

            if Confirm.ask(f"    Add these {len(review_preds)} tags?"):
                try:
                    write_tags(filepath, result["review_tags"], merge=not args.replace,
                               valid_tags=valid_tags, retired=retired)
                    written += 1
                    console.print(
                        f"    [green]✓[/green] +{', '.join(result['review_tags'])}"
                    )
                except Exception as e:
                    console.print(f"    [red]✗[/red] {e}")
            else:
                console.print(f"    [dim]Skipped[/dim]")
            console.print()

    console.print(f"\n[green]Written tags to {written} files.[/green]")


if __name__ == "__main__":
    main()
