# dnd-montage — AI detection design

Design doc for replacing the heuristic fight detector with a model that actually
*understands* the gameplay. Status: **proposal, no code yet.** Written against the
current `dnd_montage.py`.

---

## 1. Goal & current state

**Goal:** a fully automated editor — take a folder of Dark and Darker replay-buffer
clips, find the best PvP moments, cut them, and (eventually) assemble a montage.

**What we have** (`dnd_montage.py`):

- **Tier 1 — fight detection:** red crossed-swords in-combat debuff (red-pixel
  count → combat segments) ∩ Whisper voice keywords, plus the "closing fight"
  structural rule (Path A).
- **Tier 2 — class detection:** OCR the character name → class, majority vote.
- **Output:** per-fight MP4s, optional `--stitch`, HTML report, menu/market gate.

**Where it breaks (measured):** the in-combat debuff fires on **50–85 % of frames**
(vanskor-fight-1: 499 / ~593 sampled = 84 %). It cannot separate PvP from PvE
because combat is near-continuous against mobs. Voice keywords are brittle and
require you to narrate. The heuristics give **high recall, low precision** — they
know *something* is happening, not *whether it's a real player fight worth keeping*.

That precision gap is the entire reason to bring in a model.

---

## 2. The problem, decomposed

Three distinct sub-problems hide inside "find the best moments":

| # | Sub-problem | Why heuristics fail | What solves it |
|---|---|---|---|
| P1 | **PvP vs PvE** | red-X fires on both; no kill feed / nameplates | semantic vision — a geared human player looks nothing like a skeleton/goblin/cultist |
| P2 | **Montage-worthiness** (kill, clutch, funny, flashy) | no signal at all today | semantic vision + scene understanding |
| P3 | **Tight boundaries** | red-X edges are loose; voice lags | sampled-frame reasoning, refined against red-X segment edges |

P1 is the one that makes or breaks everything. It's also exactly the kind of
judgment vision models are good at and pixel-counting never could be.

---

## 3. The model question: cloud VLM vs local VLM vs trained specialist

You asked the right question: *does it have to be Claude — can't we run/train a
local specialized model?* Short answer: **yes, that's the right end-state, but you
can't start there.** Here's the honest trade-off.

### Your hardware

Two machines, two roles:

- **Mac — Apple M1 Pro, 32 GB unified.** Day-to-day box (editing, running the
  pipeline). Runs a trained student in real time; can run a 7B VLM via MLX but
  slowly; MPS training is weak.
- **PC — RTX 3060 12 GB (desktop), Ryzen 5 7600X (6c/12t), 32 GB RAM.** The ML box
  (CUDA). ✅ The 12 GB card is the comfortable case: it fits a 4-bit 7B VLM with room
  to spare, trains the CLIP-head student in seconds, and can QLoRA a 7B model
  (slowly). All heavy local work runs here.

### The three options

- **A. Cloud frontier VLM (Claude vision).** Zero training, works day one, best
  judgment on subtle P1/P2 calls. Costs per run, needs an API key + internet,
  frames leave the machine.
- **B. Local open-weights VLM** (Qwen2.5-VL, InternVL, MiniCPM-V, etc. via Ollama /
  vLLM on the PC, or MLX on the Mac). Free per run, private, offline. The realistic
  ceiling on either machine is a **7B-class model** (the 3060's 12 GB and the M1's
  32 GB both fit a quantized 7B; 32B+ is out). The **PC runs it fast** (CUDA), which
  removes the speed objection — but a 7B VLM is still **weaker than frontier** on the
  subtle PvP/PvE and "is this actually funny" calls. Hardware is no longer the limit;
  label *quality* is.
- **C. Custom-trained specialist** (your idea). A small model trained specifically
  on DnD footage. This is appealing — fast, free, private — but has a **cold-start
  problem: it needs a labeled dataset, and you have none.** You can't train a
  "best moment" detector without examples of best moments labeled as such.

### The catch with C, and the way around it

The labeling bottleneck is the whole game. Hand-labeling thousands of frames is the
expensive part — not the training, not the inference. So don't hand-label. **Use a
strong VLM as a teacher to auto-label, then distill into a small local student.**
This is a standard teacher→student / distillation pattern, and it fits your hardware
perfectly:

- Neither machine can train a frontier-quality VLM from scratch (that's the teacher
  — too big for 12–32 GB).
- The **RTX 3060 (CUDA)** comfortably trains the compact student and can even slowly
  LoRA-fine-tune a 7B VLM; the **M1 Pro** runs the trained student in real time.

So "local specialized model trained here" is realistic **as the student**, not as the
teacher. The teacher just needs to be good enough to produce clean labels once.

### Recommended path: three phases

```
Phase 1  BOOTSTRAP        VLM judge (teacher) scores candidate windows.
         (works day one)  Every verdict is also a labeled training example.
                              │  you review/correct in the HTML report
                              ▼
Phase 2  DISTILL          Train a small LOCAL classifier on the accumulated
         (once ~hundreds–   labels. Free, offline, real-time. This is your
          low-thousands     "specialized model."
          of labels)
                              │
                              ▼
Phase 3  ACTIVE-LEARNING  Local student handles confident cases; fall back to
         HYBRID            the teacher only on low-confidence windows. Labels
                           keep accruing; retrain periodically.
```

### Decided approach (local-default + one-time seed)

The chosen path: **a one-time ~$1 cloud seed → train a local student → run free and
local forever.** Concretely — the **seed teacher is Opus 4.8** over the *current*
candidate windows (best label quality, and the library is small enough that this fits
~$0.5–1.5); the **free local teacher is Qwen2.5-VL-7B** on the 3060 for new clips
going forward; the **student** is a CLIP-head trained on the PC. The seed also serves
as a calibration set to measure how much to trust the free local teacher. This gets
the best-quality foundation for the student while ending at $0 recurring cost.

The student that fits this problem and your hardware best is a **CLIP-embedding +
small head**:

1. Run frames through a frozen CLIP image encoder → one embedding per frame
   (cache them exactly like the existing `.transcripts/*.combat.json` scans).
2. Train a logistic-regression / small-MLP head on those embeddings: one head for
   PvP/PvE (P1), one regressor for montage score (P2).

Why this specifically: CLIP features already encode "armored human vs skeleton," so
a linear probe needs only **hundreds–low-thousands** of labels, trains in
**seconds–minutes on CPU**, and runs in **real time**. No GPU rental, no heavy
fine-tune. If accuracy plateaus, escalate to a fine-tuned ViT or a small
frame-sequence model later.

### Comparison

| Axis | A. Cloud VLM (Claude) | B. Local VLM (7B) | C. Trained student (CLIP+head) |
|---|---|---|---|
| Works today | ✅ instant | ✅ after model pull | ❌ needs labels first |
| Setup effort | API key + SDK | MLX/Ollama + 5–15 GB model | install + the data pipeline |
| Judgment quality (P1/P2) | highest | moderate | high *on-distribution*, after training |
| Per-run cost | ~$5–12 (Opus) / ~$2–5 (Sonnet) | $0 | $0 |
| Speed | network-bound | ~1–3 s/window on the 3060, ~5–15 s on M1 | real-time |
| Privacy / offline | frames leave machine | fully local | fully local |
| Good as | **teacher** + day-one product | budget/private teacher | **student / end-state** |

**Teacher quality caps student quality** — the student can only learn what the
teacher labeled. So if you go local-only, expect a lower ceiling than if a frontier
model does the initial labeling. The cleanest plan is: **frontier teacher for a few
runs to build a high-quality label set, then run the local student forever after.**
If privacy or zero-cost is a hard requirement from day one, swap in the local 7B VLM
as the teacher and accept a lower label-quality ceiling.

---

## 4. Architecture (model-agnostic)

Keep the cheap detectors — **demote them to a candidate generator** (high recall),
add the model as the **judge** (precision + scoring). The model never watches idle
footage, which is what keeps cost/latency sane.

```
clip.mkv
  │
  ├─ candidate windows  ◄── EXISTING: red-X combat segments + voice clusters
  │   (a few 30–60s spans, not the whole 5 min)        + closing-fight (Path A)
  │
  ├─ for each window: sample ~8–15 downscaled frames (with timestamps)
  │
  ├─ JUDGE.score(frames) ──► {is_pvp, categories, montage_score,
  │        ▲                   tight_start, tight_end, confidence, reason}
  │        │
  │   swappable backend:  claude | local_vlm | trained_head
  │
  ├─ select: keep is_pvp && montage_score ≥ threshold; refine boundaries
  │          against red-X segment edges
  │
  ├─ cut → output/<class>__<clip>__hlNN.mp4        (as today)
  │
  └─ [Phase: assembly] rank across all clips → order → stitch (+ music/titles)
```

The one new abstraction is a `Judge` interface so the backend is a config choice,
not a rewrite:

```python
class Judge(Protocol):
    def score(self, frames: list[Frame], context: WindowContext) -> Verdict: ...

# backends: ClaudeJudge, LocalVLMJudge, TrainedHeadJudge
```

Every `Verdict` is appended to a label store (sidecar JSON, same pattern as
`.transcripts/`). That store is what Phase 2 trains on — labeling is a free
byproduct of normal use.

---

## 5. Scoring rubric (your four categories)

You picked all four highlight types, so the rubric covers them. This schema is the
**single source of truth** — it's both the VLM's structured-output format and the
training-label record.

```jsonc
{
  "is_pvp": true,              // is another PLAYER involved (not just mobs)?
  "categories": [              // any of:
    "kill_win",               //  - killed a player / won the fight
    "clutch_lowhp",           //  - outnumbered, low HP, or a comeback
    "funny_fail",             //  - death to trap, whiff, trolling, meme
    "flashy_play"             //  - big hit / clean shot / slick combo, no kill needed
  ],
  "montage_score": 0,         // 0–10 overall keep-worthiness
  "tight_start": 0.0,         // refined window bounds (sec, clip-relative)
  "tight_end": 0.0,
  "confidence": 0.0,          // 0–1, drives the Phase-3 fallback
  "reason": "one line"        // human-readable, for the report + audits
}
```

Selection rule (tunable): keep if `is_pvp` **or** `funny_fail`, and
`montage_score ≥ T`. `confidence < C` routes to the teacher in Phase 3.

---

## 6. Data & labeling plan

- **Label store:** `.labels/<clip>.json` — list of `{window, frames_used, verdict,
  source: "claude"|"local_vlm"|"human"}`. Cheap, versionable, reuses the existing
  cache convention.
- **Labeling UI:** extend the existing **HTML report** — it already renders
  timelines and seeks the video. Add accept / reject / fix-category / fix-score
  controls; corrections overwrite the verdict with `source: "human"` (highest
  trust). You label by *reviewing*, which you'd do anyway.
- **Targets before training a student:** ~**300–500** labeled windows for a first
  usable PvP/PvE head; ~**1–2k** for a montage-score regressor you'd trust. You have
  ~25 clips now and add more every session, so this accrues fast.
- **Raw data is not the bottleneck** — 3.4 GB of footage already on disk yields
  plenty of frames. Labels are the bottleneck, which is exactly what Phase 1 solves.

---

## 7. Cost & performance

- **One-time seed (the only spend):** Opus 4.8 over the *current* candidate windows
  via Batches, rubric cached → ~**$0.5–1.5** (~$0.0075/window; scales with library
  size, a window cap bounds it). Everything after the seed is local and free.
- **Local 7B VLM teacher:** ~1–3 s/window on the RTX 3060 (CUDA) → a full library in
  well under an hour; ~5–15 s/window on the M1 Pro. Free, lower label quality than
  frontier.
- **Trained student:** CLIP embeddings cache once (tens of frames/sec); head trains
  in seconds–minutes — instant on the 3060, fine on the M1. Inference real-time on
  either machine. No GPU rental.
- **7B VLM LoRA fine-tune (option C2):** borderline-feasible on the 3060 (12 GB,
  QLoRA, slow) — not cloud-only anymore, but still not worth it early; the CLIP-head
  student gets ~the same outcome for far less effort.

---

## 8. Roadmap

1. **Seed.** `Judge` abstraction + `ClaudeJudge`; label current candidate windows
   with Opus via Batches (~$1); write the label store. (Detailed in §10.)
2. **Calibrate + local teacher.** Add `LocalVLMJudge` (Qwen); re-label the seed
   windows locally, measure agreement vs the Opus seed → trust level for free labels.
3. **Labeling loop.** Upgrade the HTML report into a review/correct UI.
4. **Student.** open_clip embeddings + PvP head + score head on the label store; add
   `TrainedHeadJudge`. This is the free local model.
5. **Hybrid.** Student-first, local-VLM fallback on low confidence; periodic retrain.
6. **Assembly.** Cross-clip ranking → ordering → stitch (+ music/titles).

---

## 9. Decisions

**Decided:**

- **Path:** local-default — one-time ~$1 cloud seed → train local student → run
  free/local forever. Resolves the teacher fork *and* privacy: only a small, one-time
  seed leaves the machine; everything ongoing is local.
- **Seed teacher:** Opus 4.8 over the *current* candidate windows (best labels; the
  library is small enough to fit ~$0.5–1.5). Drop to Sonnet/Haiku only if re-seeding
  a much larger library later.
- **Local teacher:** Qwen2.5-VL-7B on the 3060, for new clips going forward.
- **Student:** open_clip ViT-L/14 embeddings + sklearn heads (PvP + score).
- **Hardware:** RTX 3060 12 GB desktop confirmed — the comfortable case.

**Still open:**

- **Assembly scope** — stop at per-clip cuts (as today), or build the full cross-clip
  montage assembler. Deferred until the detector works.
- **Seed-size cap** — max windows in the one-time seed (bounds spend). Default: all
  current candidate windows (~$1).

---

## 10. Phase 1 spec — seed → student → local

**Objective:** stand up the model judge, spend the one-time ~$1 seed, train a free
local student, and run fully local afterward. No recurring cost.

**Where it runs:** all ML on the **PC** (CUDA). The label store (`.labels/`) and the
trained head (`models/`) are small files — sync them to the Mac so the student and
the existing pipeline can run there too.

**Components**

| # | Component | Status | Notes |
|---|---|---|---|
| 1 | Candidate generator | exists | `fight_windows()` — high-recall pre-filter; the judge never sees idle footage |
| 2 | Frame sampler | new (small) | ~8–10 frames/window, even spacing, downscaled ~512px JPEG (mirror `scan_combat_times`) |
| 3 | `Judge` interface | new | `score(frames, ctx) -> Verdict`; three backends below |
| 3a | `ClaudeJudge` | new | Opus via API + Batches — **seed only, one-time** |
| 3b | `LocalVLMJudge` | new | Qwen2.5-VL-7B via Ollama — free, new clips going forward |
| 3c | `TrainedHeadJudge` | new | CLIP + sklearn heads — the end-state student |
| 4 | Label store | new | `.labels/<clip>.json`: `{window, frames_used, verdict, source}` |
| 5 | Student trainer | new | open_clip ViT-L/14 → cache embeddings → train sklearn PvP + score heads → `models/` |
| 6 | Review UI | extend report | accept / reject / fix → `source:"human"` (top trust) |

**Seed plan (the only spend):** run `ClaudeJudge` (Opus 4.8) over the current
candidate windows via Batches, rubric cached. Est. **~$0.5–1.5** for today's library
(~$0.0075/window: ~10 frames × ~200 image-tokens, batch −50%; a window cap bounds it
as the library grows). The seed doubles as a **calibration set**: re-label the same
windows with the local Qwen VLM and measure agreement, so you know the free teacher's
error rate before trusting it at scale.

**Rubric / schema:** §5, unchanged — same JSON for VLM structured output and label
records.

**Install (PC):** PyTorch + CUDA, `open_clip_torch`, `scikit-learn`, Ollama +
`qwen2.5vl:7b`, `anthropic` (seed only), ffmpeg.

**Deliverables:** the new modules behind a `--judge {claude,local,trained}` flag on
`run`; the seed batch script; the trainer; the label store. Existing per-clip cutting
is unchanged.
```
