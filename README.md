# dnd-montage

Auto-edits **Dark and Darker** montages from a folder of recorded clips.

Two stages run over each clip:

- **Tier 1 — highlight detection (audio loudness).** Extracts mono PCM via
  ffmpeg, computes a windowed RMS loudness curve, flags moments above a
  per-clip percentile threshold, then pads + merges them into highlight
  windows.
- **Tier 2 — class detection.** Reads the class icon in the bottom-left of the
  screen via OpenCV template matching against `templates/`, majority-voting
  across a few sampled frames per clip (class is constant within a clip).

Each highlight is cut to its own MP4 (re-encoded for frame-accurate cuts,
libx264 CRF 18), named `<class>__<clipname>__hlNN.mp4`. It does **not**
auto-stitch by default — drop the cuts into your editor (e.g. DaVinci Resolve)
and arrange them. Pass `--stitch` to also get one concatenated file.

## Requirements

- `ffmpeg` and `ffprobe` on your `PATH`
- Python 3 with the packages in `requirements.txt`:

```sh
pip install -r requirements.txt
```

## Usage

```sh
# 1. Check the class-icon ROI box lands on the icon (run once)
python dnd_montage.py calibrate clips/some_clip.mkv

# 2. Build the icon library — once per class you play
python dnd_montage.py addtemplate clips/a_ranger_clip.mkv --name ranger

# 3. Process a whole folder
python dnd_montage.py run --in clips --out output [--stitch]
```

## Tuning

The knobs live in the `CONFIG` block at the top of `dnd_montage.py`. The ones
that matter most:

- `LOUDNESS_PERCENTILE` — higher = fewer, punchier highlights.
- `PAD_BEFORE` / `PAD_AFTER` — lead-in / tail around each detected peak.
- `ICON_ROI` — the bottom-left icon box, as frame fractions; tune with
  `calibrate`.

MKV is handled, and `probe_duration` has fallbacks for non-finalized recordings
(e.g. an OBS crash).

## Layout

```
dnd-montage/
├── dnd_montage.py    # the tool
├── clips/            # drop your raw recordings here (git-ignored)
├── output/           # cut highlights land here (git-ignored)
└── templates/        # class-icon PNGs from `addtemplate` (git-ignored)
```

## Notes

Dark and Darker has no in-game kill feed, so kill detection has to come from
other signals (audio loudness, on-screen events) rather than reading a feed.
