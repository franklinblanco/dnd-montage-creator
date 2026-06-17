# HANDOFF — continue on the PC

Instructions for the next Claude Code session (and Franklin) to continue on the
**PC** (RTX 3060 12 GB, Ryzen 5 7600X, 32 GB). Read **`DESIGN.md`** first — it's the
full plan; this file is the practical "where we are / what's next."

## TL;DR state

- **Goal:** auto-find the best PvP moments in Dark and Darker replay-buffer clips,
  cut them, eventually assemble a montage.
- **Why a model:** the heuristic detector (`dnd_montage.py`) has high recall but
  can't separate PvP from PvE — the in-combat debuff fires on 50–85% of frames
  (DESIGN.md §2).
- **Decided plan (DESIGN.md §9):** one-time ~$1 cloud seed (Opus vision labels the
  candidate windows) → train a **free local student** (CLIP embeddings + sklearn) →
  run local/free forever; a local Qwen-7B VLM labels new clips; the paid seed
  doubles as a calibration set.
- **Phase 1 = DONE** (built + unit-tested, **not run live**): `frames.py`,
  `judge.py`, `labels.py`, `seed.py`. Spec in DESIGN.md §10.
- **Phase 2 = NEXT** (the PC work): local VLM teacher + calibration, then the
  student trainer.

## Architecture (where things plug in)

```
candidate windows  (dnd_montage.fight_windows — exists, high recall)
  → frames.sample_window      (~9 frames/window, 512px JPEG)
  → judge.<Backend>.score(frames, ctx) -> Verdict
  → labels.save_verdict(...)  (.labels/<clip>.json)
  → (later) select + cut + assemble
```

`judge.Verdict` is the contract every backend emits **and** the training label:
`{is_pvp, categories[kill_win|clutch_lowhp|funny_fail|flashy_play], montage_score
0-10, tight_start, tight_end, confidence, reason}`. Keep every backend emitting it.

Backends in `judge.py`:
- `ClaudeJudge` — **DONE.** Opus via Batches, cached rubric, structured JSON. Seed only.
- `LocalVLMJudge` — **STUB.** Qwen2.5-VL-7B via Ollama. Phase 2.
- `TrainedHeadJudge` — **STUB.** CLIP + sklearn. Phase 2 (the student).

## PC setup

```sh
git clone https://github.com/franklinblanco/dnd-montage-creator.git
cd dnd-montage-creator
python -m venv .venv
. .venv/Scripts/activate          # PowerShell: .venv\Scripts\Activate.ps1 ; WSL: .venv/bin/activate
pip install -r requirements.txt   # numpy, opencv-python, pytesseract, faster-whisper, anthropic

# system tools on PATH: ffmpeg + tesseract  (choco install ffmpeg tesseract, or apt under WSL)

# Phase 2 deps (install when you start it):
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick the CUDA build for your driver
pip install open_clip_torch scikit-learn ollama
# local VLM: install Ollama (ollama.com), then:  ollama pull qwen2.5vl:7b
```

Put the **source clips** somewhere and pass `--in <dir>` — `clips/` ships empty
(videos are gitignored). `.labels/` and `models/` are gitignored too: they're your
training assets — **back them up and carry them between machines** (or just run the
seed on whichever machine holds the clips).

## Run Phase 1 (the one-time ~$1 seed) — can run on Mac or PC, no GPU needed

```sh
export ANTHROPIC_API_KEY=...                 # PowerShell: $env:ANTHROPIC_API_KEY="..."
python seed.py --in <clips-dir> --dry-run    # preview windows + cost, no API call
python seed.py --in <clips-dir>              # run it (prints actual cost from usage)
```

The first live call validates the request shape. If the API 400s on the param combo
(`output_config` effort+format + adaptive thinking inside a batch), simplify in
`ClaudeJudge._params` (drop `thinking`, or move the schema) — see DESIGN.md §10 note.
Output: `.labels/<clip>.json` records with `source:"claude"`.

## Phase 2 — build next, in order

1. **`LocalVLMJudge.score()` (judge.py).** Call Ollama: `POST {host}/api/chat` with
   `model="qwen2.5vl:7b"`, a user message carrying the rubric+context text and
   `images=[<base64 jpeg>, ...]`, and `format=judge.output_schema()` (recent Ollama
   takes a JSON schema in `format`). Parse → `Verdict` (clamp confidence, validate
   the category enum). Reuse `judge.SYSTEM` and `frames.sample_window`. Add a
   `--judge {claude,local}` flag to `seed.py` so it can label with either backend.
2. **`calibrate.py`.** Re-label the seed windows with `LocalVLMJudge` and compare to
   the existing `source:"claude"` records in `.labels/`: is_pvp agreement %,
   montage_score MAE, category overlap. Tells you how much to trust the free teacher
   before using it at scale.
3. **`train.py` (the student).** For each labeled window: sample frames → embed with
   open_clip ViT-L/14 (cache embeddings per clip, like `.transcripts/`) → mean to a
   window vector. Train sklearn LogisticRegression (is_pvp) + a regressor
   (montage_score) on the `.labels/` records; save to `models/`. Then implement
   `TrainedHeadJudge.score()` to load + predict. Target ~300–500 labels for a usable
   PvP head.
4. **Wire selection into `run`.** Add `--judge` to `dnd_montage.py run` so cutting
   keeps only `is_pvp` windows above a score threshold (student first, local-VLM
   fallback on low confidence). This replaces the brittle voice/closing-fight rule.

## Still open (decide with Franklin)

- **Assembly scope:** stop at per-clip cuts vs a full cross-clip montage (ordering,
  music, titles). DESIGN.md §9.
- **Budget:** a one-time ~$1–2 seed is approved; everything ongoing is local/free.
  Use `--max-windows` to cap spend if the library grows.

## Gotchas

- `dnd_montage.py` imports numpy/cv2/pytesseract at module load — the full
  requirements must be installed to import it (the seed pulls in `fight_windows`).
- `.transcripts/` and `.labels/` caches don't travel in git — transcripts regenerate
  per machine (need faster-whisper + the clips); labels you carry over manually.
- Model IDs: seed = `claude-opus-4-8` (best labels, fits the budget on a small
  library); local = `qwen2.5vl:7b`.
- Per-window seed cost ≈ $0.0075 (Opus via Batches); `seed.py` prints the real cost.
