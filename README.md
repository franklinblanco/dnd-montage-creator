# dnd-montage

Find the action in your **Dark and Darker** replay clips and cut a montage **fast** —
without opening and scrubbing every 5-minute clip in DaVinci.

Two parts:
1. **Detection** auto-finds the candidate fight windows in each clip, so you never
   scrub looking for where the action is.
2. **Workbench** — a browser tool to turn those into trimmed, ordered cuts and export
   them to DaVinci or a finished montage.

## Workbench (the main tool)

```sh
python workbench.py --in "C:\Users\Frank\Videos"      # opens http://localhost:8000
```

- **Pick a clip** from the bar (a badge shows how many cuts you've made on it).
- The **full clip** plays, with an **audio waveform** underneath — loud spikes are the
  action (gunshots, fights). The detector's candidate windows are drawn on the
  waveform as suggestions.
- **Drag a region** on the waveform (or *Set IN/OUT @ playhead*), pick a category,
  **+ Add cut** — as many cuts per clip as you want. Click the waveform to seek.
- All cuts collect into one **montage list on the right — drag to reorder**, retitle,
  delete, or ▶ jump to one. Everything autosaves to `.workbench/project.json`.
- **Export** three ways:
  - **Cuts → folder** — frame-accurate trimmed MP4s named in order
    (`montage/cuts/01_kill_win_*.mp4`), ready to drag into DaVinci.
  - **DaVinci timeline** — `montage/montage.fcpxml` (+ `.edl` fallback): import the
    `.fcpxml` and the whole rough cut appears on a Resolve timeline, in order, linked
    to the cut files. Add music/intro/outro there.
  - **Build montage** — stitches the cuts into `montage/montage.mp4`. Add bumpers/music
    by relaunching with `--intro intro.mkv --outro outro.mkv --music track.mp3`.

Full clips are remuxed to mp4 on demand (Resolve/Chrome can't use the raw `.mkv`) and
served with HTTP range support, so seeking is smooth. Caches live under `.workbench/`.

## Detection & per-clip cutting (`dnd_montage.py`)

The detector finds candidate fight windows from the replay-buffer structure:

- **Visual:** the red crossed-swords **in-combat debuff** (buff grid above your card)
  is detected per frame (red-pixel count) → contiguous **combat segments**.
- **Structure (Path A):** always emit the **closing fight** (last combat near the
  clip's end) plus any earlier **narrated kill** (Whisper transcript + kill keywords).
- **Class detection:** OCR your character name above the health bar → class (needs
  tesseract); **menu/market gate** skips clips with no name on screen.

```sh
python dnd_montage.py calibrate clips/some.mkv   # check the name ROI (once)
python dnd_montage.py callouts  clips/some.mkv   # combat segments + voice + windows
python dnd_montage.py run --in clips --out output [--stitch]   # cut each fight
python dnd_montage.py report --in clips --out report.html      # HTML dashboard
```

`run` cuts each fight to `output/<class>__<clip>__hlNN.mp4` (libx264 CRF 18,
frame-accurate). The workbench reuses the same detection for its suggestions.

## Requirements

```sh
# system tools on PATH:
#   ffmpeg        (required, everything)
#   tesseract     (only for dnd_montage.py class-named cutting / menu gate)
# Windows: choco install ffmpeg tesseract   ·   macOS: brew install ffmpeg tesseract

python -m venv .venv && .venv\Scripts\activate     # (bash: . .venv/Scripts/activate)
pip install -r requirements.txt                    # numpy, opencv-python, pytesseract, faster-whisper
```

The Whisper model downloads itself on first use. Transcripts and combat scans cache to
`.transcripts/`. Point `--in` at wherever the **source clips** live (they're gitignored;
`clips/` ships empty).

## Tuning

Knobs live in the `CONFIG` block of `dnd_montage.py`:

- **Visual:** `COMBAT_ROI` (buff grid above your card), `COMBAT_RED_MIN`,
  `COMBAT_MERGE_GAP`, `COMBAT_MIN_SEC`.
- **Structure / Path A:** `CLOSING_FIGHT_SEC`, `PAD_BEFORE` / `PAD_AFTER`, `MAX_HL_SEC`.
- **Voice:** `KILL_WORDS` / `ENGAGE_WORDS` (match how you talk), `WHISPER_MODEL`.
- **Class:** `NAME_OVERRIDES`, `NAME_ROI`, `NAME_MATCH_CUTOFF`.

## What about an automatic PvP detector?

DnD has no kill feed and no nameplates, so the detector has high recall but can't tell
**PvP from PvE** on its own (the in-combat debuff fires on mobs too). We explored an AI
judge to close that gap (`judge.py`, `seed.py`, `calibrate.py`; design in
[DESIGN.md](DESIGN.md)):

- A free **local Qwen2.5-VL-7B** judge (via Ollama) was tested against hand labels and
  **could not separate PvP from PvE** (≈0 recall; no prompt/CoT/resolution variant beat
  a trivial baseline — see `tune_local.py`). Not usable.
- A one-time paid **Opus vision seed** (`seed.py --judge claude`) is the approach that
  *would* work and doubles as a calibration set, but needs API credit. It remains the
  open path if you want to automate selection later.

Until then, selection is **human-driven in the workbench** — which, given replay clips
yield only ~1–2 windows each, is fast and gives perfect results.

## Notes

- Class names in clip *titles* are usually the opponents fought, not your class.
- `.labels/` (hand labels), `.workbench/`, `.transcripts/`, and `montage/` are
  generated and gitignored.
