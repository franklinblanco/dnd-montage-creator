# dnd-montage

Auto-edits **Dark and Darker** montages from a folder of recorded clips.

Two stages run over each clip:

- **Tier 1 — highlight detection (audio loudness).** Extracts mono PCM via
  ffmpeg, computes a windowed RMS loudness curve, flags moments above a
  per-clip percentile threshold, then pads + merges them into highlight
  windows.
- **Tier 2 — class detection (player-name OCR).** Reads your character name
  from above the health bar (bottom-center) with Tesseract, fuzzy-matches it
  against a `NAME_TO_CLASS` roster, and majority-votes across a few sampled
  frames per clip (your character is constant within a clip).

Each highlight is cut to its own MP4 (re-encoded for frame-accurate cuts,
libx264 CRF 18), named `<class>__<clipname>__hlNN.mp4`. It does **not**
auto-stitch by default — drop the cuts into your editor (e.g. DaVinci Resolve)
and arrange them. Pass `--stitch` to also get one concatenated file.

## Requirements

- `ffmpeg`, `ffprobe`, and `tesseract` on your `PATH`:

  ```sh
  brew install ffmpeg tesseract
  ```

- Python 3 with the packages in `requirements.txt`:

  ```sh
  pip install -r requirements.txt
  ```

- Fill in `NAME_TO_CLASS` at the top of `dnd_montage.py` — map each of *your*
  character names to its class. Clips whose name matches none are `unknown`.

## Usage

```sh
# 1. Check the name ROI box lands on your character name (run once)
python dnd_montage.py calibrate clips/some_clip.mkv

# 2. (Optional) sanity-check the OCR read + matched class on a clip
python dnd_montage.py readname clips/some_clip.mkv

# 3. Process a whole folder
python dnd_montage.py run --in clips --out output [--stitch]
```

## Tuning

The knobs live in the `CONFIG` block at the top of `dnd_montage.py`:

- `LOUDNESS_PERCENTILE` — higher = fewer, punchier highlights.
- `PAD_BEFORE` / `PAD_AFTER` — lead-in / tail around each detected peak.
- `NAME_TO_CLASS` — your `name → class` roster.
- `NAME_ROI` — the name-line box, as frame fractions; tune with `calibrate`.
- `NAME_MATCH_CUTOFF` — min fuzzy-match score (0–1) to accept a roster name.

MKV is handled, and `probe_duration` has fallbacks for non-finalized recordings
(e.g. an OBS crash).

## Layout

```
dnd-montage/
├── dnd_montage.py    # the tool
├── clips/            # drop your raw recordings here (git-ignored)
└── output/           # cut highlights land here (git-ignored)
```

## Notes

- Dark and Darker has no in-game kill feed, so kill detection has to come from
  other signals (audio loudness, on-screen events) rather than reading a feed.
- Class names in clip *titles* (e.g. `…rogue-fighter`) are typically the
  *opponents* fought, for sorting — not the player's own class.
