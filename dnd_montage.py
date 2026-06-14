#!/usr/bin/env python3
"""
dnd_montage.py — Dark and Darker montage helper

Two jobs:
  1) TIER 1  -> find the PvP fights. Boundaries come from the VISUAL in-combat
                debuff (a red crossed-swords icon in the buff grid above your
                card) detected per frame and merged into combat segments; VOICE
                callouts (Whisper) then keep the segments that are real PvP
                fights (vs PvE mobs) and rank them by kills.
  2) TIER 2  -> read which class you were playing from your character name
                (shown above the health bar), via OCR.

It does NOT auto-stitch by default. It cuts each highlight into its own file,
named with the detected class, so you can drag them into DaVinci Resolve and
arrange them yourself. (Pass --stitch to also get one concatenated file.)

----------------------------------------------------------------------
SETUP (once)
----------------------------------------------------------------------
  - Install FFmpeg + Tesseract and put `ffmpeg`/`ffprobe`/`tesseract` on PATH.
      macOS:  brew install ffmpeg tesseract
  - pip install -r requirements.txt   (numpy, opencv-python, pytesseract,
    faster-whisper). The Whisper model downloads itself on first use.
  - Class is read from your character name (names embed their class). Only add
    a line to NAME_OVERRIDES below for a character whose name does NOT.
  - Tune the KILL_WORDS / ENGAGE_WORDS lists to match how YOU talk in fights.

----------------------------------------------------------------------
WORKFLOW
----------------------------------------------------------------------
  Step 1 — check the name box lands on your character name (do this once):
      python dnd_montage.py calibrate "clips/some_clip.mkv"
      -> writes _calib_frame.png (full frame with a red NAME_ROI box drawn),
         _calib_crop.png (what's inside the box), and _calib_ocr.png (what
         OCR actually sees). The red box should sit over the name above your
         health bar. If it's off, tweak NAME_ROI below and run again.

  Step 2 — sanity-check the reads on a clip (optional):
      python dnd_montage.py readname "clips/some_clip.mkv"   # class from name
      python dnd_montage.py callouts "clips/some_clip.mkv"   # transcript + fights

  Step 3 — process a whole folder:
      python dnd_montage.py run --in clips --out output
      -> output/ranger__some_clip__hl01.mp4, etc.
         Add --stitch to also get output/_montage_all.mp4

Tuning: everything you'll want to touch lives in the CONFIG block below. The
knobs that matter most are COMBAT_RED_MIN / COMBAT_MERGE_GAP (visual combat
boundaries), the KILL_WORDS / ENGAGE_WORDS lists and PVP_MIN_WEIGHT (which
combat counts as a PvP fight), and PAD_BEFORE / PAD_AFTER (lead-in / tail).
"""

import argparse
import os
import re
import json
import subprocess
import sys
import glob
import shutil
import tempfile
import difflib

import numpy as np
import cv2
import pytesseract

# ======================================================================
# CONFIG — tune these
# ======================================================================

# --- Fight detection: visual combat boundaries ∩ voice significance (Tier 1) ---
# Boundaries come from the VISUAL "in-combat" debuff (a red crossed-swords icon
# in the fixed buff grid above the player's card, bottom-left). It fires on every
# damage instance and fades, so we detect it as red-pixel density per frame and
# merge contiguous combat into one segment — true fight boundaries that survive
# quiet stretches and merge chained fights / 3rd-parties.
COMBAT_ROI        = (0.004, 0.798, 0.078, 0.858)  # buff grid above MY card
COMBAT_RED_MIN    = 20    # saturated-red px in the ROI to call a frame "in combat"
COMBAT_FPS        = 2.0   # frames/sec sampled for the combat scan
COMBAT_MERGE_GAP  = 8.0   # bridge combat gaps up to this many sec (one fight)
COMBAT_MIN_SEC    = 4.0   # ignore combat blips shorter than this (stray mob hit)

# The red X also fires on PvE mobs, so VOICE decides which combat is a real PvP
# fight worth keeping, and ranks it. Keywords match as case-insensitive
# substrings ("dead" covers "he's dead", "one dead", "ranger dead").
WHISPER_MODEL     = "small.en"  # faster-whisper model: tiny/base/small/medium .en
KILL_WORDS   = ["dead", "got him", "got one", "killed"]
# (no "running"/"run" — they matched casual speech like "running in circles")
ENGAGE_WORDS = ["hit him", "hit", "shooting", "shoot", "push",
                "players", "people", "trail", "behind", "coming", "reset"]
SCOUT_WORDS  = ["geared", "kitted", "naked", "ranger", "rogue", "fighter",
                "wizard", "sorcerer", "druid", "barbarian", "bard", "cleric",
                "warlock"]
KILL_W, ENGAGE_W, SCOUT_W = 3.0, 1.0, 0.3   # per-keyword weights

# Path A — exploit the structure: you clip via replay buffer right after a PvP
# fight, so every clip has a payoff fight near the END. We always emit that
# "closing fight" (last combat window near the clip end — no voice needed, so
# it survives un-narrated fights), plus any earlier NARRATED KILL as its own
# fight. PvE-vs-PvP is sidestepped: we already know the clip contains a fight.
CLOSING_FIGHT_SEC = 50.0  # length of the closing-fight window (back from the end)
VOICE_CLUSTER_GAP = 20.0  # group voice callouts within this many sec into a fight
VOICE_MARGIN      = 8.0   # a kill-cluster must coincide with combat within this
FIGHT_EXTEND      = 15.0  # extend a kill window this far into adjacent combat
PAD_BEFORE        = 4.0   # min lead-in before a fight
PAD_AFTER         = 6.0   # min tail after a fight (keeps the kill/aftermath)
MIN_HL_SEC        = 4.0   # drop fight windows shorter than this
MAX_HL_SEC        = 90.0  # hard cap on a single fight window

# --- Class detection via player-name OCR (Tier 2) ---
# Your character name is drawn above the health bar (bottom-center). We OCR it
# and read the class straight from the name: your character names embed their
# class (e.g. a name ending in "RANGER" -> ranger), so we fuzzy-match the class
# names below against the OCR'd text and majority-vote across sampled frames.
CLASSES = ["fighter", "ranger", "rogue", "wizard", "sorcerer",
           "druid", "barbarian", "bard", "cleric", "warlock"]

# Exceptions: character names that DON'T embed their class. Map name -> class.
NAME_OVERRIDES = {
    "FinallyBalanced": "druid",
}

# ROI of the name line as frame fractions (x0, y0, x1, y1). Tune with `calibrate`.
NAME_ROI          = (0.43, 0.892, 0.59, 0.932)
NAME_SCAN_STEP    = 3.0    # seconds between name-OCR samples across the clip
NAME_MATCH_CUTOFF = 0.72   # min fuzzy-match ratio (0-1) to accept a class

# Menu/market gate: a clip (or a stretch of it) with no character name on screen
# isn't gameplay and isn't worth clipping. If the name never shows near a
# highlight, we skip it; if it never shows at all, we skip the whole clip.
MARKET_CTX_SEC    = 30.0   # look this far around a highlight for the name

# --- Output encoding ---
# Re-encoding gives frame-accurate cuts (stream-copy only cuts on keyframes).
VIDEO_CODEC       = "libx264"
CRF               = "18"
PRESET            = "veryfast"
AUDIO_CODEC       = "aac"

# ======================================================================
# ffmpeg / ffprobe helpers
# ======================================================================

def run_quiet(cmd):
    """Run a command, return CompletedProcess, swallow ffmpeg's chatter."""
    return subprocess.run(cmd, capture_output=True)

def probe_duration(path):
    """Duration in seconds, with fallbacks for MKVs that lack a header duration
    (common when a recording wasn't finalized cleanly, e.g. an OBS crash)."""
    # 1) container-level duration
    out = run_quiet([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1", path
    ]).stdout.decode().strip()
    try:
        d = float(out)
        if d > 0:
            return d
    except ValueError:
        pass

    # 2) video stream duration
    out = run_quiet([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=nokey=1:noprint_wrappers=1", path
    ]).stdout.decode().strip()
    try:
        d = float(out)
        if d > 0:
            return d
    except ValueError:
        pass

    # 3) last resort: decode and read the real end timestamp from ffmpeg
    err = run_quiet(["ffmpeg", "-i", path, "-map", "0:v:0",
                     "-f", "null", "-"]).stderr.decode(errors="ignore")
    last = None
    for tok in err.split("time="):
        ts = tok[:11].strip()
        parts = ts.split(":")
        if len(parts) == 3:
            try:
                last = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            except ValueError:
                pass
    if last and last > 0:
        return last

    raise RuntimeError(f"Could not read duration of {path}")

def grab_frame(path, t, out_png):
    """Extract a single frame at time t (seconds) to a PNG."""
    run_quiet(["ffmpeg", "-ss", f"{t:.3f}", "-i", path,
               "-frames:v", "1", "-y", out_png])
    return out_png

# ======================================================================
# TIER 1 — highlight detection via voice callouts (Whisper transcription)
# ======================================================================

TRANSCRIBE_CACHE = ".transcripts"
_WHISPER = None

def _whisper():
    """Lazily load the Whisper model once (reused across clips in a run)."""
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel
        _WHISPER = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _WHISPER

def transcribe(path):
    """Return [(start, end, text)] for the clip's commentary, cached to
    .transcripts/<clip>.json and invalidated when the source file changes."""
    os.makedirs(TRANSCRIBE_CACHE, exist_ok=True)
    cache = os.path.join(TRANSCRIBE_CACHE, os.path.basename(path) + ".json")
    mtime = os.path.getmtime(path)
    if os.path.exists(cache):
        with open(cache) as fh:
            data = json.load(fh)
        if data.get("src_mtime") == mtime and data.get("model") == WHISPER_MODEL:
            return data["segments"]

    wav = "_asr.wav"
    run_quiet(["ffmpeg", "-i", path, "-ac", "1", "-ar", "16000",
               "-f", "wav", "-y", "-v", "quiet", wav])
    segments, _ = _whisper().transcribe(wav, vad_filter=True, language="en")
    segs = [[round(s.start, 3), round(s.end, 3), s.text.strip()]
            for s in segments]
    if os.path.exists(wav):
        os.remove(wav)
    with open(cache, "w") as fh:
        json.dump({"src_mtime": mtime, "model": WHISPER_MODEL, "segments": segs}, fh)
    return segs

def score_text(text):
    """Return (weight, has_kill) for a transcript line from its callout words."""
    t = text.lower()
    weight, kill = 0.0, False
    for k in KILL_WORDS:
        if k in t:
            weight += KILL_W
            kill = True
    for k in ENGAGE_WORDS:
        if k in t:
            weight += ENGAGE_W
    for k in SCOUT_WORDS:
        if k in t:
            weight += SCOUT_W
    return weight, kill

def combat_red_px(frame):
    """Saturated-red pixels in the buff grid above the player's card. The combat
    debuff is the only red icon there (other buffs are white/gold/blue)."""
    roi = crop_roi(frame, COMBAT_ROI)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 110, 70), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 110, 70), (180, 255, 255))
    return int(((m1 | m2) > 0).sum())

def scan_combat_times(path):
    """Times (sec) where the in-combat debuff was on, sampled at COMBAT_FPS.
    Cached to .transcripts/<clip>.combat.json, invalidated by file/param change."""
    os.makedirs(TRANSCRIBE_CACHE, exist_ok=True)
    cache = os.path.join(TRANSCRIBE_CACHE, os.path.basename(path) + ".combat.json")
    key = [os.path.getmtime(path), list(COMBAT_ROI), COMBAT_RED_MIN, COMBAT_FPS]
    if os.path.exists(cache):
        with open(cache) as fh:
            data = json.load(fh)
        if data.get("key") == key:
            return data["hits"]

    tmpdir = tempfile.mkdtemp(prefix="dndcombat_")
    try:
        run_quiet(["ffmpeg", "-i", path, "-vf", f"fps={COMBAT_FPS}", "-q:v", "3",
                   os.path.join(tmpdir, "f_%05d.jpg"), "-v", "quiet"])
        hits = []
        for i, fp in enumerate(sorted(glob.glob(os.path.join(tmpdir, "f_*.jpg")))):
            frame = cv2.imread(fp)
            if frame is not None and combat_red_px(frame) >= COMBAT_RED_MIN:
                hits.append(round(i / COMBAT_FPS, 2))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    with open(cache, "w") as fh:
        json.dump({"key": key, "hits": hits}, fh)
    return hits

def combat_segments(path, duration):
    """Merged combat spans [(start, end)] — these are ALL combat (PvP and PvE)."""
    segs = []
    for t in scan_combat_times(path):
        if segs and t - segs[-1][1] <= COMBAT_MERGE_GAP:
            segs[-1][1] = t
        else:
            segs.append([t, t])
    return [(s, e) for s, e in segs if e - s >= COMBAT_MIN_SEC]

def voice_pvp_clusters(path):
    """Group PvP voice callouts into fight cores: time-ordered clusters of
    (start, end, weight, has_kill)."""
    voice = [(s, e, *score_text(text)) for s, e, text in transcribe(path)]
    voice = [(s, e, w, k) for s, e, w, k in voice if w > 0]
    clusters = []
    for s, e, w, k in voice:
        if clusters and s - clusters[-1]["end"] <= VOICE_CLUSTER_GAP:
            c = clusters[-1]
            c["end"] = max(c["end"], e)
            c["weight"] += w
            c["kill"] = c["kill"] or k
        else:
            clusters.append({"start": s, "end": e, "weight": w, "kill": k})
    return clusters

def fight_windows(path, duration):
    """Path A. Always emit the closing fight (last combat near the clip end),
    plus any earlier narrated kill as its own window. Returns time-ordered
    (start, end, weight, has_kill)."""
    segs = combat_segments(path, duration)
    if not segs:
        return []                      # no combat at all -> menu/market clip
    clusters = voice_pvp_clusters(path)

    def voice_near(s, e):
        w = sum(c["weight"] for c in clusters if c["start"] <= e and c["end"] >= s)
        k = any(c["kill"] for c in clusters if c["start"] <= e and c["end"] >= s)
        return round(w, 1), k

    windows = []
    # (1) the closing fight — the payoff, anchored at the last combat
    cs0, ce0 = segs[-1]
    e = min(duration, ce0 + PAD_AFTER)
    s = max(0.0, max(cs0, ce0 - CLOSING_FIGHT_SEC) - PAD_BEFORE)
    w, k = voice_near(s, e)
    windows.append([s, e, w, k])

    # (2) earlier narrated kills (separate PvP fights worth keeping)
    for c in clusters:
        if not c["kill"]:
            continue
        seg = next(((a, b) for a, b in segs if a - VOICE_MARGIN <= c["end"]
                    and b + VOICE_MARGIN >= c["start"]), None)
        if seg is None:
            continue
        a, b = seg
        ks = max(0.0, max(a, c["start"] - FIGHT_EXTEND) - PAD_BEFORE)
        ke = min(duration, min(b, c["end"] + FIGHT_EXTEND) + PAD_AFTER)
        if ke > windows[0][0] and ks < windows[0][1]:
            continue               # overlaps the closing fight — already covered
        kw, _ = voice_near(ks, ke)
        windows.append([ks, ke, kw, True])

    # merge any overlapping windows (adjacent kills can produce overlaps)
    merged = []
    for s, e, w, k in sorted(windows):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2] = max(merged[-1][2], w)
            merged[-1][3] = merged[-1][3] or k
        else:
            merged.append([s, e, w, k])

    out = []
    for s, e, w, k in merged:
        if e - s < MIN_HL_SEC:
            continue
        if e - s > MAX_HL_SEC:
            s = e - MAX_HL_SEC     # keep the end (the payoff), trim the front
        out.append((round(s, 3), round(e, 3), w, k))
    return out

# ======================================================================
# TIER 2 — class detection via player-name OCR
# ======================================================================

def roi_pixels(w, h, roi):
    x0, y0, x1, y1 = roi
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)

def crop_roi(img, roi):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi_pixels(w, h, roi)
    return img[y0:y1, x0:x1]

def preprocess_name(crop):
    """Upscale + grayscale + Otsu threshold so the light serif text reads well.
    Returns black-text-on-white, which is what Tesseract prefers."""
    big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th

def ocr_name(crop):
    """OCR a single name line."""
    return pytesseract.image_to_string(
        preprocess_name(crop), config="--psm 7").strip()

def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _best_align(needle, hay):
    """Best SequenceMatcher ratio of `needle` aligned anywhere in `hay`, so
    leading/trailing OCR junk around the bit we care about doesn't sink it."""
    if not needle or not hay:
        return 0.0
    if len(hay) <= len(needle):
        return difflib.SequenceMatcher(None, needle, hay).ratio()
    return max(
        difflib.SequenceMatcher(None, needle, hay[i:i + len(needle)]).ratio()
        for i in range(0, len(hay) - len(needle) + 1)
    )

def match_class(text):
    """Return (class, score) for OCR'd name `text`. Check name overrides first
    (their names don't embed the class), then look for an embedded class name."""
    t = _norm(text)
    if not t:
        return None, 0.0
    best_cls, best_score = None, 0.0
    for name, cls in NAME_OVERRIDES.items():
        score = _best_align(_norm(name), t)
        if score > best_score:
            best_cls, best_score = cls, score
    for cls in CLASSES:
        score = _best_align(cls, t)
        if score > best_score:
            best_cls, best_score = cls, score
    return best_cls, best_score

def scan_names(path, duration):
    """One pass over the clip: OCR the name line every NAME_SCAN_STEP seconds.
    Returns a list of (t, class, score) — reused for both class voting and the
    menu/market gate, so we only decode frames once."""
    scan = []
    tmp = "_scan_frame.png"
    for t in np.arange(NAME_SCAN_STEP, duration, NAME_SCAN_STEP):
        grab_frame(path, float(t), tmp)
        frame = cv2.imread(tmp)
        if frame is None:
            continue
        cls, score = match_class(ocr_name(crop_roi(frame, NAME_ROI)))
        scan.append((float(t), cls, score))
    if os.path.exists(tmp):
        os.remove(tmp)
    return scan

def vote_class(scan):
    """Majority-vote the class over the frames where the name read clearly."""
    votes, best = {}, 0.0
    for _, cls, score in scan:
        if cls and score >= NAME_MATCH_CUTOFF:
            votes[cls] = votes.get(cls, 0) + 1
            best = max(best, score)
    if not votes:
        return "unknown", 0.0
    return max(votes, key=votes.get), best

def name_hit_times(scan):
    """Times where the character name read clearly (= we were in gameplay)."""
    return [t for t, cls, score in scan if cls and score >= NAME_MATCH_CUTOFF]

def is_gameplay(hit_times, s, e):
    """A highlight is gameplay if the name shows anywhere within MARKET_CTX_SEC
    of it. Name reads are sparse, so we look at a window, not single frames; a
    contiguous no-name stretch (menu/market, or post-death) won't qualify."""
    return any(s - MARKET_CTX_SEC <= t <= e + MARKET_CTX_SEC for t in hit_times)

# ======================================================================
# Cutting
# ======================================================================

def cut(path, start, end, out_path):
    # -ss before -i = fast seek; re-encode for frame-accurate boundaries
    run_quiet([
        "ffmpeg", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", path,
        "-c:v", VIDEO_CODEC, "-crf", CRF, "-preset", PRESET,
        "-c:a", AUDIO_CODEC, "-y", "-v", "quiet", out_path
    ])

def concat(clip_paths, out_path):
    listfile = "_concat_list.txt"
    with open(listfile, "w") as fh:
        for c in clip_paths:
            fh.write(f"file '{os.path.abspath(c)}'\n")
    run_quiet(["ffmpeg", "-f", "concat", "-safe", "0", "-i", listfile,
               "-c", "copy", "-y", "-v", "quiet", out_path])
    os.remove(listfile)

# ======================================================================
# Modes
# ======================================================================

def mode_calibrate(args):
    dur = probe_duration(args.clip)
    grab_frame(args.clip, dur / 2.0, "_calib_frame.png")
    frame = cv2.imread("_calib_frame.png")
    if frame is None:
        sys.exit("Could not read a frame from that clip.")
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = roi_pixels(w, h, NAME_ROI)
    crop = frame[y0:y1, x0:x1].copy()
    cv2.imwrite("_calib_crop.png", crop)
    cv2.imwrite("_calib_ocr.png", preprocess_name(crop))
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite("_calib_frame.png", frame)
    print(f"Frame is {w}x{h}. NAME_ROI pixels: x {x0}-{x1}, y {y0}-{y1}")
    print(f"OCR of this frame: '{ocr_name(crop)}'")
    print("Open _calib_frame.png — the red box should sit over the name")
    print("above your health bar. _calib_ocr.png shows what OCR sees.")
    print("If it's off, edit NAME_ROI and run calibrate again.")

def mode_readname(args):
    dur = probe_duration(args.clip)
    tmp = "_read_frame.png"
    scan = []
    for t in np.arange(NAME_SCAN_STEP, dur, NAME_SCAN_STEP):
        grab_frame(args.clip, float(t), tmp)
        frame = cv2.imread(tmp)
        if frame is None:
            continue
        raw = ocr_name(crop_roi(frame, NAME_ROI))
        cls, score = match_class(raw)
        ok = bool(cls) and score >= NAME_MATCH_CUTOFF
        scan.append((t, cls, score))
        print(f"  t={t:>5.0f}s {'#' if ok else ' '} ocr='{raw}'"
              f"  -> {cls or '-'} ({score:.2f})")
    if os.path.exists(tmp):
        os.remove(tmp)

    hits = len(name_hit_times(scan))
    cls, score = vote_class(scan)
    print(f"\nName visible on {hits}/{len(scan)} sampled frame(s).")
    if cls == "unknown":
        print("Verdict: no name found — this clip would be treated as menu/market.")
    else:
        print(f"Verdict: class={cls} (conf {score:.2f})")

def mode_callouts(args):
    """Diagnostic: show visual combat segments, the voice callouts near them,
    and the fused fight windows — use this to tune detection."""
    dur = probe_duration(args.clip)
    segs = combat_segments(args.clip, dur)
    print(f"duration {dur:.0f}s\n\nVisual combat segments (red-X debuff):")
    if not segs:
        print("  (none)")
    for cs, ce in segs:
        print(f"  [{cs:6.1f}-{ce:6.1f}]  ({ce-cs:.0f}s)")

    print("\nVoice callouts:")
    for s, e, text in transcribe(args.clip):
        w, kill = score_text(text)
        if w > 0:
            print(f"  [{s:6.1f}-{e:6.1f}] ({w:.1f}{' KILL' if kill else ''}) {text}")

    print("\nFused fight windows:")
    fights = fight_windows(args.clip, dur)
    if not fights:
        print("  (none)")
    for i, (s, e, w, kill) in enumerate(fights, 1):
        tag = " KILL" if kill else ""
        print(f"  hl{i:02d}  [{s:.1f}s - {e:.1f}s]  PvP-score {w:.1f}{tag}")

def _mmss(t):
    return f"{int(t)//60}:{int(t)%60:02d}"

def analyze_clip(path):
    """Gather everything the report needs for one clip."""
    dur = probe_duration(path)
    scan = scan_names(path, dur)
    cls, _ = vote_class(scan)
    combat = combat_segments(path, dur)
    callouts = []
    for s, e, text in transcribe(path):
        w, kill = score_text(text)
        if w > 0:
            callouts.append({"s": s, "e": e, "w": w, "kill": kill, "text": text})
    fights = fight_windows(path, dur)
    return {"path": path, "name": os.path.splitext(os.path.basename(path))[0],
            "dur": dur, "cls": cls, "combat": combat,
            "callouts": callouts, "fights": fights}

def mode_report(args):
    """Build a self-contained HTML dashboard for a folder of clips: per clip, a
    player + a timeline (orange=in combat/likely PvE, red=PvP fight, ticks=kills)
    and a clickable kill/fight list that seeks the video."""
    import html as _html
    from urllib.parse import quote

    clips = []
    for ext in ("mp4", "mkv", "mov", "MP4", "MKV", "MOV"):
        clips += glob.glob(os.path.join(args.in_dir, f"*.{ext}"))
    clips = sorted(set(clips))
    if not clips:
        sys.exit(f"No video files found in {args.in_dir}")

    sections = []
    for clip in clips:
        try:
            a = analyze_clip(clip)
        except RuntimeError as e:
            print(f"  ! skipping {os.path.basename(clip)}: {e}")
            continue
        dur = a["dur"] or 1.0
        pct = lambda t: 100.0 * t / dur
        bands = "".join(
            f'<div class="band combat" style="left:{pct(s):.2f}%;'
            f'width:{pct(e-s):.2f}%"></div>' for s, e in a["combat"])
        bands += "".join(
            f'<div class="band fight" style="left:{pct(s):.2f}%;'
            f'width:{pct(e-s):.2f}%" title="fight {_mmss(s)}-{_mmss(e)}'
            f'{ " KILL" if k else ""}"></div>' for s, e, w, k in a["fights"])
        ticks = "".join(
            f'<div class="tick{" kill" if c["kill"] else ""}" '
            f'style="left:{pct(c["s"]):.2f}%" title="{_html.escape(c["text"])}"></div>'
            for c in a["callouts"])
        items = []
        for i, (s, e, w, k) in enumerate(a["fights"], 1):
            tag = " <b>KILL</b>" if k else ""
            items.append(f'<li data-t="{s:.1f}">🎬 fight hl{i:02d} '
                         f'[{_mmss(s)}–{_mmss(e)}] score {w}{tag}</li>')
        for c in a["callouts"]:
            if c["kill"]:
                items.append(f'<li data-t="{c["s"]:.1f}">💀 {_mmss(c["s"])} '
                             f'— "{_html.escape(c["text"])}"</li>')
        src = "file://" + quote(os.path.abspath(clip))
        sections.append(f"""
  <section>
    <h2>{_html.escape(a["name"])} <small>({a["cls"]}, {_mmss(dur)})</small></h2>
    <video controls preload="none" src="{src}"></video>
    <div class="timeline" data-dur="{dur:.2f}" onclick="tl(event,this)">
      {bands}{ticks}
    </div>
    <ul class="markers">{''.join(items) or '<li>(no fights detected)</li>'}</ul>
  </section>""")

    doc = f"""<!doctype html><meta charset="utf-8">
<title>dnd-montage report — {_html.escape(os.path.basename(args.in_dir.rstrip('/')))}</title>
<style>
  body{{font:14px system-ui;background:#1a1a1a;color:#ddd;margin:24px;max-width:960px}}
  section{{border-top:1px solid #333;padding:16px 0}}
  h2{{margin:0 0 8px}} small{{color:#888;font-weight:400}}
  video{{max-width:480px;background:#000;display:block;margin-bottom:8px}}
  .timeline{{position:relative;height:34px;background:#222;border-radius:4px;cursor:pointer;overflow:hidden}}
  .band{{position:absolute;top:0;height:100%}}
  .band.combat{{background:rgba(230,150,40,.45)}}
  .band.fight{{background:rgba(220,40,40,.55);border:1px solid #f33}}
  .tick{{position:absolute;top:0;width:2px;height:100%;background:#888}}
  .tick.kill{{background:#ff3b3b;width:3px}}
  .markers{{list-style:none;padding:0;margin:8px 0 0}}
  .markers li{{padding:3px 6px;cursor:pointer;border-radius:3px}}
  .markers li:hover{{background:#2c2c2c}}
  .legend span{{margin-right:16px}} .sw{{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-1px;margin-right:4px}}
</style>
<h1>dnd-montage report</h1>
<p class="legend">
  <span><i class="sw" style="background:rgba(230,150,40,.7)"></i>in combat (likely PvE)</span>
  <span><i class="sw" style="background:rgba(220,40,40,.8)"></i>PvP fight</span>
  <span><i class="sw" style="background:#ff3b3b"></i>kill callout</span>
</p>
{''.join(sections)}
<script>
document.querySelectorAll('.markers li[data-t]').forEach(li=>li.onclick=()=>{{const v=li.closest('section').querySelector('video');v.currentTime=+li.dataset.t;v.play();}});
function tl(e,el){{const r=el.getBoundingClientRect();const dur=+el.dataset.dur;const v=el.closest('section').querySelector('video');v.currentTime=(e.clientX-r.left)/r.width*dur;v.play();}}
</script>"""

    with open(args.out, "w") as fh:
        fh.write(doc)
    print(f"Wrote {args.out}  ({len(sections)} clip(s)). Open it in a browser.")

def mode_run(args):
    os.makedirs(args.out, exist_ok=True)

    clips = []
    for ext in ("mp4", "mkv", "mov", "MP4", "MKV", "MOV"):
        clips += glob.glob(os.path.join(args.in_dir, f"*.{ext}"))
    clips = sorted(set(clips))
    if not clips:
        sys.exit(f"No video files found in {args.in_dir}")

    all_outputs = []
    for clip in clips:
        base = os.path.splitext(os.path.basename(clip))[0]
        try:
            dur = probe_duration(clip)
        except RuntimeError as e:
            print(f"  ! skipping {base}: {e}")
            continue

        scan = scan_names(clip, dur)
        cls, score = vote_class(scan)
        if cls == "unknown":
            print(f"{base}: no character name on screen — assuming menu/market, "
                  f"skipping.")
            continue

        hit_times = name_hit_times(scan)
        fights = fight_windows(clip, dur)
        kept = [(s, e, w, k) for s, e, w, k in fights
                if is_gameplay(hit_times, s, e)]
        skipped = len(fights) - len(kept)
        msg = f"{base}: class={cls} (conf {score:.2f}), {len(kept)} fight(s)"
        if skipped:
            msg += f", {skipped} skipped (menu/market)"
        print(msg)

        for idx, (s, e, w, kill) in enumerate(kept, 1):
            out_name = f"{cls}__{base}__hl{idx:02d}.mp4"
            out_path = os.path.join(args.out, out_name)
            cut(clip, s, e, out_path)
            all_outputs.append(out_path)
            tag = " KILL" if kill else ""
            print(f"    -> {out_name}  [{s:.1f}s - {e:.1f}s]  score {w:.1f}{tag}")

    if args.stitch and all_outputs:
        montage = os.path.join(args.out, "_montage_all.mp4")
        concat(all_outputs, montage)
        print(f"\nStitched montage -> {montage}")

    print(f"\nDone. {len(all_outputs)} clip(s) written to {args.out}/")

# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(description="Dark and Darker montage helper")
    sub = p.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("calibrate", help="check where the name ROI lands")
    c.add_argument("clip")
    c.set_defaults(func=mode_calibrate)

    rn = sub.add_parser("readname", help="show OCR name reads + matched class")
    rn.add_argument("clip")
    rn.set_defaults(func=mode_readname)

    co = sub.add_parser("callouts", help="show transcript, keyword scores, windows")
    co.add_argument("clip")
    co.set_defaults(func=mode_callouts)

    rp = sub.add_parser("report", help="build an HTML dashboard (timelines/kills)")
    rp.add_argument("--in", dest="in_dir", required=True)
    rp.add_argument("--out", dest="out", default="report.html")
    rp.set_defaults(func=mode_report)

    r = sub.add_parser("run", help="process a folder of clips")
    r.add_argument("--in", dest="in_dir", default="clips")
    r.add_argument("--out", dest="out", default="output")
    r.add_argument("--stitch", action="store_true",
                   help="also concatenate all highlights into one file")
    r.set_defaults(func=mode_run)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
