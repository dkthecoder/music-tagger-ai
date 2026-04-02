#!/usr/bin/env python3
"""Extract audio embeddings from a tagged music library.

Uses the Discogs-EffNet model to extract 1280-dimensional embeddings
from each track, paired with existing genre tags from file metadata.
Output is saved as .npy files for training.

Usage:
    python scripts/extract_embeddings.py /path/to/tagged/library
    python scripts/extract_embeddings.py /path/to/library --output data/embeddings
    python scripts/extract_embeddings.py /path/to/library --tag-type subgenre
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs
import mutagen
from mutagen.flac import FLAC
from mutagen.id3 import ID3

console = Console()

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "discogs-effnet-bs64-1.pb"
TAXONOMY_PATH = Path(__file__).parent.parent / "taxonomy.yaml"

AUDIO_EXTENSIONS = {".flac", ".mp3"}


def load_taxonomy():
    """Load taxonomy for tag validation."""
    with open(TAXONOMY_PATH) as f:
        return yaml.safe_load(f)


def get_genre_tags(filepath: Path) -> list[str]:
    """Read genre tags from a music file."""
    suffix = filepath.suffix.lower()

    try:
        if suffix == ".flac":
            audio = FLAC(str(filepath))
            # FLAC stores multi-value as repeated keys
            tags = audio.tags
            if tags is None:
                return []
            genres = []
            for key, value in tags:
                if key.upper() == "GENRE":
                    genres.append(value)
            return genres

        elif suffix == ".mp3":
            audio = ID3(str(filepath))
            tcon = audio.get("TCON")
            if tcon:
                # ID3 TCON can contain multiple genres separated by /
                raw = str(tcon)
                return [g.strip() for g in raw.split("/") if g.strip()]
            return []

    except Exception:
        return []

    return []


def extract_embedding(filepath: Path) -> np.ndarray | None:
    """Extract 1280-dim embedding from a single track."""
    try:
        audio = MonoLoader(
            filename=str(filepath), sampleRate=16000, resampleQuality=4
        )()
        embeddings = TensorflowPredictEffnetDiscogs(
            graphFilename=str(MODEL_PATH),
            output="PartitionedCall:1",  # 1280-dim embeddings
        )(audio)
        # Average across all frames to get a single vector
        return np.mean(embeddings, axis=0)
    except Exception as e:
        console.print(f"[red]  Error extracting {filepath.name}: {e}[/red]")
        return None


def find_tagged_files(path: Path) -> list[Path]:
    """Find all audio files that have genre tags."""
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(path.rglob(f"*{ext}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from tagged music library"
    )
    parser.add_argument("path", type=Path, help="Path to tagged music library")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Output directory for embeddings (default: data/)",
    )
    parser.add_argument(
        "--tag-type",
        choices=["all", "subgenre", "regional", "mood"],
        default="all",
        help="Which tag types to extract (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of files to process (0 = no limit)",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        console.print(f"[red]Model not found at {MODEL_PATH}[/red]")
        sys.exit(1)

    if not args.path.exists():
        console.print(f"[red]Path not found: {args.path}[/red]")
        sys.exit(1)

    taxonomy = load_taxonomy()

    # Collect all known tags for filtering
    known_tags = set()
    known_tags.update(taxonomy.get("base_genres", []))
    for sg_list in taxonomy.get("subgenres", {}).values():
        if isinstance(sg_list, list):
            known_tags.update(sg_list)
    known_tags.update(taxonomy.get("regional", []))
    known_tags.update(taxonomy.get("scene", []))
    known_tags.update(taxonomy.get("style", []))
    known_tags.update(taxonomy.get("mood", []))

    # Find files
    all_files = find_tagged_files(args.path)
    console.print(f"Found {len(all_files)} audio files")

    # Filter to files with tags
    tagged_files = []
    file_tags = {}
    for f in all_files:
        tags = get_genre_tags(f)
        if tags:
            tagged_files.append(f)
            file_tags[str(f)] = tags

    console.print(f"  {len(tagged_files)} have genre tags")

    if args.limit > 0:
        tagged_files = tagged_files[: args.limit]
        console.print(f"  Limited to {len(tagged_files)} files")

    if not tagged_files:
        console.print("[red]No tagged files found.[/red]")
        sys.exit(1)

    # Collect unique tags
    all_tags = set()
    for tags in file_tags.values():
        all_tags.update(tags)
    console.print(f"  {len(all_tags)} unique genre tags found")
    console.print()

    # Extract embeddings
    embeddings = []
    labels = []
    filepaths = []
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting embeddings", total=len(tagged_files))

        for filepath in tagged_files:
            progress.update(task, description=f"[cyan]{filepath.name[:40]}[/cyan]")
            embedding = extract_embedding(filepath)
            if embedding is not None:
                embeddings.append(embedding)
                labels.append(file_tags[str(filepath)])
                filepaths.append(str(filepath))
            else:
                skipped += 1
            progress.advance(task)

    console.print()
    console.print(f"Extracted {len(embeddings)} embeddings ({skipped} skipped)")

    # Save
    args.output.mkdir(parents=True, exist_ok=True)

    embeddings_array = np.array(embeddings)
    np.save(args.output / "embeddings.npy", embeddings_array)

    metadata = {
        "filepaths": filepaths,
        "labels": labels,
        "embedding_dim": int(embeddings_array.shape[1]) if len(embeddings) > 0 else 0,
        "num_tracks": len(embeddings),
        "unique_tags": sorted(all_tags),
    }
    with open(args.output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    console.print(f"[green]Saved to {args.output}/[/green]")
    console.print(f"  embeddings.npy: {embeddings_array.shape}")
    console.print(f"  metadata.json: {len(filepaths)} tracks, {len(all_tags)} tags")

    # Print tag distribution
    tag_counts = {}
    for tags in labels:
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    console.print("\nTag distribution (top 20):")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:20]:
        bar = "█" * min(count // 5, 40)
        console.print(f"  {tag:25s} {count:4d} {bar}")


if __name__ == "__main__":
    main()
