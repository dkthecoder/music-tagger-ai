# Genre Taxonomy

This document defines the tagging system used by music-tagger-ai. It's designed as a layered classification system where songs receive multiple tags across different dimensions, enabling filtering by genre, region, era, or vibe without needing playlists.

## Design Principles

1. **Tags describe what music sounds like and feels like** — not just what a streaming service labels it
2. **Multiple tags per song** — a song can be `Hip-Hop/Rap` + `Afro Trap` + `Deutsch` + `Energetic`
3. **Apple Music conventions** for base genre names where possible
4. **Compound genres are single names** — `Hip-Hop/Rap` is ONE genre (not hip-hop + rap). Don't create false compounds like `Hip-Hop/Drill`
5. **Never use standalone `Hip-Hop`** — always `Hip-Hop/Rap`. Same for `R&B` → always `R&B/Soul`

## Tag Types

### Base Genres
Primary genre — standalone names using Apple Music conventions:

`Hip-Hop/Rap`, `R&B/Soul`, `Pop`, `Electronic`, `House`, `Dance`, `Dancehall`, `Reggae`, `Soul`, `Bollywood`, `Amapiano`, `Metal`, `Rock`, `Hard Rock`, `Funk`, `Disco`, `Electro`, `Techno`, `Nu Disco`, `Nu Metal`, `Bhangra`, `Indian Pop`, `Alternative`

### Subgenres
Layers on top of a base genre:

| Subgenre | Description |
|----------|-------------|
| `Drill` | Dark, aggressive, sliding 808s |
| `Trap` | Heavy 808s, hi-hats, modern hip-hop production |
| `Grime` | UK electronic hip-hop, 140 BPM |
| `G-House` | Ghetto house / bass house |
| `Boom-Bap` | Classic 90s hip-hop production |
| `Afro Trap` | European rap with West African-influenced beats (MHD, BILLA JOE) |
| `Arab Trap` | European rap with Arab/Turkish/Balkan-influenced beats (Summer Cem, Azet) |
| `Afrobeats` | West African pop (Nigerian/Ghanaian origin) |
| `Afro House` | African-influenced house music |
| `Dark Rap` | Dark, atmospheric hip-hop (Night Lovell, Ramirez) |
| `Neo Soul` | Modern soul (Snoh Aalegra, Sabrina Claudio) |
| `Rapcore` | Rap + rock/metal (Limp Bizkit, Bloodywood) |
| `Garage` | UK garage / 2-step |
| `Lo-Fi` | Lo-fi hip-hop / chill beats |
| `Phonk` | Memphis-influenced dark electronic |
| `Brazilian Phonk` | Brazilian variation of phonk |
| `PluggnB` | Dreamy trap + 90s R&B (emerging 2025+) |

### Regional / Country Tags
Origin or scene of the artist:

`US`, `UK`, `Deutsch`, `Svenska`, `Swiss`, `Norsk`, `Turkish`, `Romanian`, `Russian`, `Punjabi`, `Spanish`, `ZA`, `Canada`, `Latin`, `Brazilian`, `East Coast`, `West Coast`, `Southern`, `Bay Area`

### Scene Tags
Distinct music scenes crossing genres and regions:

`Urban Asian` — South Asian diaspora music scene blending hip-hop, R&B, garage, bhangra, and Bollywood

### Style / Era Markers
`Old School`, `Remix`, `Old School Remix`, `Soundtrack`

### Mood Tags (AI-detected)
Future implementation — detected from audio analysis:

`Energetic`, `Dark`, `Melodic`, `Aggressive`, `Chill`, `Party`, `Cinematic`, `Nostalgic`

## Custom Subgenres (Transfer Learning)

These subgenres don't exist in standard classification systems. The AI learns to recognise them from examples in the user's tagged library:

| Custom Tag | Training Examples | What makes it distinct |
|-----------|-------------------|----------------------|
| `Afro Trap` | ~60 tracks (BILLA JOE, Luciano, LieVin) | West African rhythms + European rap, danceable |
| `Arab Trap` | ~50 tracks (Summer Cem, Azet, Zuna, Fousy) | Oriental melodies, Arab/Turkish hooks over trap |
| `Dark Rap` | ~55 tracks (Night Lovell, Ramirez) | Atmospheric, dark, cloud rap influenced |
| `Neo Soul` | ~80 tracks (Snoh Aalegra, Sabrina Claudio, Alina Baraz) | Dreamy R&B, cinematic soul |
| `Urban Asian` | ~500 tracks (DJ Sanj, RDB, Rishi Rich) | UK Desi fusion scene |

## Layering Examples

| Song type | Genre entries |
|-----------|--------------|
| German Afro Trap | `Hip-Hop/Rap`, `Afro Trap`, `Deutsch` |
| Balkan trap with dancehall | `Hip-Hop/Rap`, `Arab Trap`, `Dancehall`, `Deutsch` |
| UK garage R&B | `Hip-Hop/Rap`, `R&B/Soul`, `UK`, `Garage` |
| Swedish neo soul | `R&B/Soul`, `Neo Soul`, `Svenska` |
| Bollywood (single composer) | `Bollywood`, album = film name, albumartist = composer |
| Multicultural European street rap | Tag song-by-song based on featured artists and audio analysis |
