# music-tagger-ai

AI-powered music library tagger that goes beyond standard genre classification. Uses Essentia's deep learning models + transfer learning to classify niche subgenres that mainstream tools can't recognise.

## The Problem

Streaming services and auto-taggers classify music into broad categories like "Hip-Hop" or "Pop". But real music is more nuanced:

- **Afro Trap** — European rap with West African-influenced beats (MHD, BILLA JOE)
- **Arab Trap** — Turkish/Arab/Balkan-influenced trap (Summer Cem, Azet)
- **Urban Asian** — UK-born South Asian fusion scene (Rishi Rich, DJ Sanj)
- **Dancehall-Rap** — Caribbean-influenced European rap (Loredana, Ramriddlz)

No existing tool can tell the difference between these. This one can — because it learns from YOUR library.

## How It Works

1. **Pre-trained analysis** — Essentia's Discogs model classifies 400+ standard genres with BPM detection
2. **Custom classification** — Transfer learning on your tagged library teaches the model YOUR subgenres
3. **Tag suggestions** — Analyses untagged music and suggests tags with confidence scores
4. **You decide** — Review suggestions before they're written to files

```
Your tagged library (7000+ files)
         ↓
Essentia extracts audio embeddings
         ↓
Train classifier on YOUR custom tags
         ↓
Run on new/untagged music
         ↓
"This sounds like Afro Trap (87% confidence)"
         ↓
Review → Approve → Tags written to files
```

## Features

- Audio-based analysis (not metadata lookups)
- Pre-trained Discogs 400 genre/style classification
- BPM detection
- Custom subgenre classification via transfer learning
- Batch processing for entire libraries
- Dry-run mode (preview before writing)
- Supports FLAC and MP3

## Genre Taxonomy

The project uses a layered tagging system where songs get multiple tags across different dimensions:

| Layer | Purpose | Examples |
|-------|---------|---------|
| **Base genre** | Primary genre (Apple Music conventions) | `Hip-Hop/Rap`, `R&B/Soul`, `Pop`, `Electronic` |
| **Subgenre** | Specific classification | `Drill`, `Trap`, `Afro Trap`, `Arab Trap`, `Neo Soul` |
| **Regional** | Artist origin/scene | `US`, `UK`, `Deutsch`, `Svenska`, `ZA` |
| **Scene** | Cultural music scenes | `Urban Asian` |
| **Style** | Era/format markers | `Old School`, `Remix` |
| **Mood** | Energy/vibe (planned) | `Energetic`, `Dark`, `Melodic`, `Aggressive` |

See [docs/taxonomy.md](docs/taxonomy.md) for the full genre system.

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/music-tagger-ai.git
cd music-tagger-ai

# Create virtual environment (requires Python 3.12+)
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Analyse a folder with pre-trained models
python scripts/analyse.py /path/to/music

# 2. Extract embeddings from your tagged library (for transfer learning)
python scripts/extract_embeddings.py /path/to/tagged/library

# 3. Train custom classifier on your tags
python scripts/train_classifier.py

# 4. Predict tags for untagged music (dry-run preview)
python scripts/predict_tags.py /path/to/untagged/music

# 5. Write approved tags to files
python scripts/predict_tags.py /path/to/untagged/music --apply
```

## Scripts

| Script | What it does |
|--------|-------------|
| `scripts/analyse.py` | Analyse tracks with the pre-trained Discogs model — top genre/style predictions + BPM. Table or JSON output |
| `scripts/extract_embeddings.py` | Scan a tagged library, extract 1280-dim audio embeddings, pair with existing genre tags. Saves `.npy` + `.json` |
| `scripts/train_classifier.py` | Train a multi-label classifier (OneVsRest SGD) on extracted embeddings. Per-tag F1 scores, cross-validation, saves `.pkl` |
| `scripts/predict_tags.py` | Run the trained classifier on new/untagged music. Shows suggestions with confidence scores. Dry-run by default, `--apply` to write tags to files |

## Tag Writing

When writing tags to files (`predict_tags.py --apply`), the script enforces the taxonomy conventions from `taxonomy.yaml`:

- **Retired tag mapping** — automatically corrects legacy tags (e.g. `Hip-Hop` → `Hip-Hop/Rap`, `R&B` → `R&B/Soul`, `Desi` → `Urban Asian`)
- **False compound splitting** — `Hip-Hop/Drill` becomes two separate tags: `Hip-Hop/Rap` + `Drill`
- **Deduplication** — never writes the same tag twice, preserves order
- **Mixed-case normalisation** — strips both `genre` and `GENRE` Vorbis keys, rewrites as uppercase `GENRE`
- **Existing tag cleanup** — when merging with existing tags on a file, retired tags already on the file are also corrected
- **Unknown tag warning** — tags not in the taxonomy are still written but flagged with a warning
- **FLAC** — writes proper multi-value Vorbis entries (one `GENRE=` per tag, not delimited strings)
- **MP3** — writes to ID3v2 TCON frame

## The Cultural Story

This project was born from trying to organise a 7000+ track music library spanning:

- **European multicultural rap** — German, Swedish, Norwegian rappers with African, Arab, Turkish, Balkan heritage creating genres that don't exist in standard classification systems
- **South Asian diaspora music** — UK-born "Urban Asian" scene blending hip-hop, R&B, garage, and bhangra
- **Bollywood restructuring** — Moving from film-name folders to composer-based organisation
- **Classic genre cleanup** — Standardising R&B/Soul, Hip-Hop/Rap, fixing hundreds of typos and inconsistencies

Read the full story in the [LinkedIn article series](#articles).

## Articles

1. [My Music Library Was a Mess — So I Built a System](#)
2. [Beyond "Hip-Hop/Rap" — Mapping Europe's Multicultural Music Scene](#)
3. [Teaching AI to Hear What Spotify Can't](#)
4. [From 400 Discogs Genres to My Own: Transfer Learning for Music](#)
5. [Open-Sourcing the Music Tagger: What I Learned](#)

## Tech Stack

- [Essentia](https://essentia.upf.edu/) — Audio analysis and pre-trained ML models
- [scikit-learn](https://scikit-learn.org/) — Custom classifier training (OneVsRest SGD)
- [TensorFlow](https://www.tensorflow.org/) — Backend for Essentia's deep learning models
- [mutagen](https://mutagen.readthedocs.io/) — Audio metadata reading/writing (FLAC Vorbis + MP3 ID3)
- [librosa](https://librosa.org/) — Audio feature extraction
- [Rich](https://rich.readthedocs.io/) — Terminal output formatting
- Python 3.12+

## License

MIT

## Acknowledgements

- [Music Technology Group, UPF Barcelona](https://www.upf.edu/web/mtg) — Essentia library
- [Discogs](https://www.discogs.com/) — Genre taxonomy used in pre-trained models
