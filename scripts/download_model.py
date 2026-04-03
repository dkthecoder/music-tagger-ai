#!/usr/bin/env python3
"""Download the pre-trained Discogs-EffNet model.

Downloads the model and metadata files from Essentia's model repository.
Run this once after cloning the repo.

Usage:
    python scripts/download_model.py
"""

import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

FILES = {
    "discogs-effnet-bs64-1.pb": (
        "https://essentia.upf.edu/models/music-style-classification/"
        "discogs-effnet/discogs-effnet-bs64-1.pb"
    ),
    "discogs-effnet-bs64-1.json": (
        "https://essentia.upf.edu/models/music-style-classification/"
        "discogs-effnet/discogs-effnet-bs64-1.json"
    ),
}


def download(url: str, dest: Path):
    """Download a file with progress."""
    print(f"  Downloading {dest.name}...")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Essentia Discogs-EffNet model...\n")

    for filename, url in FILES.items():
        dest = MODEL_DIR / filename
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  {filename} already exists ({size_mb:.1f} MB) — skipping")
            continue
        try:
            download(url, dest)
        except Exception as e:
            print(f"\n  Error downloading {filename}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\nDone. Model files saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
