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

from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs
import mutagen
from mutagen.flac import FLAC
from mutagen.id3 import ID3

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
        print(f"  ERROR: {filepath.name}: {e}", file=sys.stderr, flush=True)
        return None


def find_tagged_files(path: Path) -> list[Path]:
    """Find all audio files that have genre tags."""
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(path.rglob(f"*{ext}"))
    return sorted(files)


def _save_checkpoint(embeddings, labels, filepaths, all_tags, output_dir):
    """Save incremental checkpoint (won't overwrite final files)."""
    embeddings_array = np.array(embeddings)
    np.save(output_dir / "embeddings_checkpoint.npy", embeddings_array)
    metadata = {
        "filepaths": filepaths,
        "labels": labels,
        "embedding_dim": int(embeddings_array.shape[1]) if len(embeddings) > 0 else 0,
        "num_tracks": len(embeddings),
        "unique_tags": sorted(all_tags),
    }
    with open(output_dir / "metadata_checkpoint.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Checkpoint saved: {len(embeddings)} tracks", flush=True)


def _save_final(embeddings, labels, filepaths, all_tags, output_dir):
    """Save final output files."""
    embeddings_array = np.array(embeddings)
    np.save(output_dir / "embeddings.npy", embeddings_array)
    metadata = {
        "filepaths": filepaths,
        "labels": labels,
        "embedding_dim": int(embeddings_array.shape[1]) if len(embeddings) > 0 else 0,
        "num_tracks": len(embeddings),
        "unique_tags": sorted(all_tags),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


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
        print(f"Error: Model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    if not args.path.exists():
        print(f"Error: Path not found: {args.path}", file=sys.stderr)
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
    print(f"Found {len(all_files)} audio files", flush=True)

    # Filter to files with tags
    tagged_files = []
    file_tags = {}
    for f in all_files:
        tags = get_genre_tags(f)
        if tags:
            tagged_files.append(f)
            file_tags[str(f)] = tags

    print(f"  {len(tagged_files)} have genre tags", flush=True)

    if args.limit > 0:
        tagged_files = tagged_files[: args.limit]
        print(f"  Limited to {len(tagged_files)} files", flush=True)

    if not tagged_files:
        print("No tagged files found.", flush=True)
        sys.exit(1)

    # Collect unique tags
    all_tags = set()
    for tags in file_tags.values():
        all_tags.update(tags)
    print(f"  {len(all_tags)} unique genre tags found\n", flush=True)

    # Extract embeddings with incremental saves
    embeddings = []
    labels = []
    filepaths = []
    skipped = 0
    save_every = 100  # save checkpoint every N tracks

    args.output.mkdir(parents=True, exist_ok=True)
    total = len(tagged_files)

    # Resume from existing checkpoint if available
    checkpoint_path = args.output / "embeddings_checkpoint.npy"
    checkpoint_meta = args.output / "metadata_checkpoint.json"
    start_idx = 0

    if checkpoint_path.exists() and checkpoint_meta.exists():
        existing_emb = np.load(checkpoint_path)
        with open(checkpoint_meta) as f:
            existing_meta = json.load(f)
        embeddings = list(existing_emb)
        labels = existing_meta["labels"]
        filepaths = existing_meta["filepaths"]
        start_idx = len(filepaths)
        print(f"Resuming from checkpoint: {start_idx}/{total} tracks already done", flush=True)

    for i, filepath in enumerate(tagged_files[start_idx:], start=start_idx + 1):
        # Log to stdout so background tasks can see progress
        if i % 50 == 0 or i == start_idx + 1:
            pct = i * 100 // total
            print(f"[{i}/{total}] ({pct}%) {filepath.name}", flush=True)

        embedding = extract_embedding(filepath)
        if embedding is not None:
            embeddings.append(embedding)
            labels.append(file_tags[str(filepath)])
            filepaths.append(str(filepath))
        else:
            skipped += 1
            print(f"  SKIPPED: {filepath.name}", flush=True)

        # Incremental save every N tracks
        if i % save_every == 0:
            _save_checkpoint(embeddings, labels, filepaths, all_tags, args.output)

    print(f"\nExtracted {len(embeddings)} embeddings ({skipped} skipped)", flush=True)

    # Final save (overwrites checkpoint with final files)
    _save_final(embeddings, labels, filepaths, all_tags, args.output)

    # Clean up checkpoint files
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if checkpoint_meta.exists():
        checkpoint_meta.unlink()

    print(f"Saved to {args.output}/", flush=True)
    print(f"  embeddings.npy: ({len(embeddings)}, 1280)", flush=True)
    print(f"  metadata.json: {len(filepaths)} tracks, {len(all_tags)} tags", flush=True)

    # Print tag distribution
    tag_counts = {}
    for tags in labels:
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print("\nTag distribution (top 20):", flush=True)
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:20]:
        bar = "█" * min(count // 5, 40)
        print(f"  {tag:25s} {count:4d} {bar}", flush=True)


if __name__ == "__main__":
    main()
