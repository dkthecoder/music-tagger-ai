#!/usr/bin/env python3
"""Analyse music files using Essentia's pre-trained Discogs model.

Outputs genre/style predictions, BPM, and mood estimates for each track.

Usage:
    python scripts/analyse.py /path/to/music
    python scripts/analyse.py /path/to/music --top 10 --format json
    python scripts/analyse.py /path/to/track.flac
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from essentia.standard import MonoLoader, RhythmExtractor2013
from essentia.standard import TensorflowPredictEffnetDiscogs

console = Console()

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "discogs-effnet-bs64-1.pb"
LABELS_PATH = MODEL_DIR / "discogs-effnet-bs64-1.json"
TAXONOMY_PATH = Path(__file__).parent.parent / "taxonomy.yaml"

AUDIO_EXTENSIONS = {".flac", ".mp3", ".wav", ".ogg", ".m4a"}


def load_labels():
    """Load Discogs genre/style labels from the model metadata."""
    with open(LABELS_PATH) as f:
        meta = json.load(f)
    classes = meta.get("classes")
    if not classes:
        # Fallback: look for metadata key
        for key in meta:
            if isinstance(meta[key], list) and len(meta[key]) > 100:
                classes = meta[key]
                break
    if not classes:
        console.print("[red]Could not find class labels in model metadata[/red]")
        sys.exit(1)
    return classes


def load_taxonomy():
    """Load our custom taxonomy for mapping predictions."""
    with open(TAXONOMY_PATH) as f:
        return yaml.safe_load(f)


def find_audio_files(path: Path):
    """Find all audio files in a directory (or return the file itself)."""
    if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
        return [path]
    if path.is_dir():
        files = []
        for ext in AUDIO_EXTENSIONS:
            files.extend(path.rglob(f"*{ext}"))
        return sorted(files)
    return []


def analyse_track(filepath: Path, top_n: int = 5):
    """Analyse a single track and return predictions."""
    # Load audio at 16kHz mono (required by Discogs model)
    audio = MonoLoader(filename=str(filepath), sampleRate=16000, resampleQuality=4)()

    # Get genre/style predictions (400 classes)
    predictions = TensorflowPredictEffnetDiscogs(
        graphFilename=str(MODEL_PATH),
        output="PartitionedCall:0",
    )(audio)

    # Average predictions across all frames
    avg_preds = np.mean(predictions, axis=0)

    # Load labels
    labels = load_labels()

    # Get top predictions
    top_indices = np.argsort(avg_preds)[::-1][:top_n]
    top_predictions = [
        {"label": labels[i], "confidence": float(avg_preds[i])}
        for i in top_indices
    ]

    # BPM detection
    try:
        audio_44k = MonoLoader(filename=str(filepath), sampleRate=44100)()
        bpm, *_ = RhythmExtractor2013(method="multifeature")(audio_44k)
        bpm = round(float(bpm))
    except Exception:
        bpm = None

    return {
        "file": str(filepath),
        "filename": filepath.name,
        "predictions": top_predictions,
        "bpm": bpm,
    }


def print_results(results, top_n: int = 5):
    """Print results as a rich table."""
    for result in results:
        table = Table(title=result["filename"], show_header=True)
        table.add_column("Rank", style="dim", width=4)
        table.add_column("Genre/Style", style="cyan")
        table.add_column("Confidence", justify="right", style="green")

        for i, pred in enumerate(result["predictions"][:top_n], 1):
            conf = f"{pred['confidence']:.1%}"
            table.add_row(str(i), pred["label"], conf)

        if result["bpm"]:
            table.add_row("", f"BPM: {result['bpm']}", "", style="yellow")

        console.print(table)
        console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyse music with Essentia's Discogs model"
    )
    parser.add_argument("path", type=Path, help="File or folder to analyse")
    parser.add_argument(
        "--top", type=int, default=5, help="Number of top predictions (default: 5)"
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        console.print(f"[red]Model not found at {MODEL_PATH}[/red]")
        console.print("Download it with:")
        console.print(
            "  curl -L https://essentia.upf.edu/models/classification-heads/"
            "discogs-effnet/discogs-effnet-bs64-1.pb -o models/discogs-effnet-bs64-1.pb"
        )
        sys.exit(1)

    files = find_audio_files(args.path)
    if not files:
        console.print(f"[red]No audio files found at {args.path}[/red]")
        sys.exit(1)

    console.print(f"Analysing {len(files)} track(s)...\n")

    results = []
    for i, filepath in enumerate(files, 1):
        console.print(f"[dim][{i}/{len(files)}] {filepath.name}[/dim]")
        try:
            result = analyse_track(filepath, top_n=args.top)
            results.append(result)
        except Exception as e:
            console.print(f"[red]  Error: {e}[/red]")

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        console.print()
        print_results(results, top_n=args.top)

    console.print(f"[green]Done. Analysed {len(results)}/{len(files)} tracks.[/green]")


if __name__ == "__main__":
    main()
