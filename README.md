# dnd-montage

Auto-finds the **PvP fights** in **Dark and Darker** clips and cuts each to its
own MP4 for editing.

## How it works

- **Tier 1 — fight detection (visual + voice).** You clip via a replay buffer
  right after a fight, so every clip contains a PvP fight, usually near the end.
  - *Visual:* the red crossed-swords **in-combat debuff** (in the buff grid above
    your card) is detected per frame (red-pixel count) → contiguous **combat
    segments**. Reliable, but fires on PvE mobs too.
  - *Structure (Path A):* always emit the **closing fight** — the last combat
    window near the clip's end (no voice needed) — plus any earlier **narrated
    kill** (Whisper transcript + kill keywords) as its own window.
- **Tier 2 — class detection.** OCR your character name above the health bar,
  read the class embedded in it (e.g. `…RANGER` → ranger), majority-vote.
- **Menu/market gate.** Clips with no character name on screen are skipped.

Each fight is cut to `output/<class>__<clip>__hlNN.mp4` (libx264 CRF 18,
frame-accurate). Pass `--stitch` to also get one concatenated file.

## Requirements

```sh
brew install ffmpeg tesseract
pip install -r requirements.txt          # numpy, opencv-python, pytesseract, faster-whisper
```

The Whisper model downloads itself on first use. Transcripts and combat scans
are cached under `.transcripts/`.

## Usage

```sh
# Check the name ROI lands on your character name (once)
python dnd_montage.py calibrate clips/some_clip.mkv

# Diagnostics for a single clip
python dnd_montage.py readname clips/some_clip.mkv    # class from name
python dnd_montage.py callouts clips/some_clip.mkv    # combat segments + voice + windows

# Process a whole folder
python dnd_montage.py run --in clips --out output [--stitch]

# Build an HTML dashboard (timelines, kills, in-combat vs PvP) for a folder
python dnd_montage.py report --in clips --out report.html
```

## AI seed labeling (Phase 1)

The heuristic detector finds *candidate* fight windows but can't tell PvP from PvE
(the in-combat debuff fires on mobs too). `seed.py` labels those windows once with
Claude vision — PvP or not, montage score 0–10, plus highlight categories — and
writes a training set to `.labels/` for a free local model to learn from later.
This one-time step is the only paid part; see [DESIGN.md](DESIGN.md) for the plan.

```sh
pip install anthropic
export ANTHROPIC_API_KEY=...
python seed.py --in clips --dry-run    # preview windows + ~cost, no API call
python seed.py --in clips              # run the one-time seed (~$1 on a small library)
```

Needs the **source** clips present (point `--in` at them). Runs on any machine with
ffmpeg + the requirements — no GPU. The local model and training come in later phases.

## Tuning

Knobs live in the `CONFIG` block of `dnd_montage.py`:

- **Visual:** `COMBAT_ROI` (buff grid above your card), `COMBAT_RED_MIN`
  (red px to count a frame as in-combat), `COMBAT_MERGE_GAP`, `COMBAT_MIN_SEC`.
- **Structure / Path A:** `CLOSING_FIGHT_SEC` (length of the end-of-clip fight
  window), `PAD_BEFORE` / `PAD_AFTER`, `MAX_HL_SEC`.
- **Voice:** `KILL_WORDS` / `ENGAGE_WORDS` (match how you talk), `WHISPER_MODEL`.
- **Class:** `NAME_OVERRIDES`, `NAME_ROI`, `NAME_MATCH_CUTOFF`.

## Notes

- DnD has no kill feed and no enemy nameplates; combat is near-continuous (PvE),
  and health loss doesn't separate PvP from PvE — hence the replay-buffer
  "closing fight" structure (Path A) rather than trying to classify PvP directly.
- Class names in clip *titles* are usually the opponents fought, not your class.
