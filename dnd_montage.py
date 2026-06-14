#!/usr/bin/env python3
"""
dnd_montage.py — Dark and Darker montage helper

Two jobs:
  1) TIER 1  -> find the loud/action moments in each clip (audio loudness).
  2) TIER 2  -> read which class you were playing from your character name
                (shown above the health bar), via OCR + a name->class map.

It does NOT auto-stitch by default. It cuts each highlight into its own file,
named with the detected class, so you can drag them into DaVinci Resolve and
arrange them yourself. (Pass --stitch to also get one concatenated file.)

----------------------------------------------------------------------
SETUP (once)
----------------------------------------------------------------------
  - Install FFmpeg + Tesseract and put `ffmpeg`/`ffprobe`/`tesseract` on PATH.
      macOS:  brew install ffmpeg tesseract
  - pip install -r requirements.txt   (numpy, opencv-python, pytesseract)
  - Class is read from your character name (names embed their class). Only add
    a line to NAME_OVERRIDES below for a character whose name does NOT.

----------------------------------------------------------------------
WORKFLOW
----------------------------------------------------------------------
  Step 1 — check the name box lands on your character name (do this once):
      python dnd_montage.py calibrate "clips/some_clip.mkv"
      -> writes _calib_frame.png (full frame with a red NAME_ROI box drawn),
         _calib_crop.png (what's inside the box), and _calib_ocr.png (what
         OCR actually sees). The red box should sit over the name above your
         health bar. If it's off, tweak NAME_ROI below and run again.

  Step 2 — sanity-check the read on a clip (optional):
      python dnd_montage.py readname "clips/some_clip.mkv"
      -> prints the raw OCR per sampled frame and the class it matched.

  Step 3 — process a whole folder:
      python dnd_montage.py run --in clips --out output
      -> output/ranger__some_clip__hl01.mp4, etc.
         Add --stitch to also get output/_montage_all.mp4

Tuning: everything you'll want to touch lives in the CONFIG block below.
The two knobs that matter most are LOUDNESS_PERCENTILE (higher = pickier)
and PAD_BEFORE / PAD_AFTER (how much lead-in / tail each highlight gets).
"""

import argparse
import os
import re
import subprocess
import sys
import glob
import difflib

import numpy as np
import cv2
import pytesseract

# ======================================================================
# CONFIG — tune these
# ======================================================================

# --- Audio / highlight detection (Tier 1) ---
AUDIO_SR          = 8000   # Hz to downsample audio to for analysis (plenty for loudness)
WIN_SEC           = 0.40   # length of each loudness measurement window
HOP_SEC           = 0.10   # how far the window slides each step
SMOOTH_SEC        = 0.50   # moving-average smoothing over the loudness curve

LOUDNESS_PERCENTILE = 95.0 # a moment counts as "action" if louder than this %
                           #   of the clip. Higher = fewer, punchier highlights.
ABS_FLOOR_DB      = -40.0  # never flag anything quieter than this (kills silence)

PAD_BEFORE        = 3.0    # seconds of lead-in before a detected peak
PAD_AFTER         = 5.0    # seconds of tail after it
MERGE_GAP         = 6.0    # merge two highlights if they're within this many sec
MIN_HL_SEC        = 2.0    # drop highlights shorter than this
MAX_HL_SEC        = 30.0   # hard cap on a single highlight's length

# End-bias: clips come from a replay buffer, so the payoff is almost always
# near the END (you clip after a fight / while looting). Favor the tail, and
# treat the very start skeptically so early ambient loudness doesn't crowd in.
# (Action at the start usually means a double-clip or a long fight — still
# caught, but it has to be genuinely loud to clear the raised bar.)
END_ZONE_SEC      = 90.0   # the final this-many seconds are the "payoff zone"
END_BOOST_DB      = 4.0    # lower the loudness bar by this much in the end zone
START_GUARD_SEC   = 30.0   # the first this-many seconds are treated skeptically
START_GUARD_DB    = 5.0    # raise the loudness bar by this much in the start guard

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
# TIER 1 — loudness-based highlight detection
# ======================================================================

def load_audio(path):
    """Decode to mono PCM float in [-1, 1] via ffmpeg piped through numpy."""
    raw = run_quiet([
        "ffmpeg", "-i", path, "-ac", "1", "-ar", str(AUDIO_SR),
        "-f", "s16le", "-v", "quiet", "-"
    ]).stdout
    if not raw:
        raise RuntimeError(f"No audio decoded from {path}")
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

def loudness_curve(samples):
    """Return (times, dB) — windowed RMS loudness over the clip."""
    win_n = int(AUDIO_SR * WIN_SEC)
    hop_n = int(AUDIO_SR * HOP_SEC)
    times, db = [], []
    for start in range(0, max(1, len(samples) - win_n), hop_n):
        seg = samples[start:start + win_n]
        rms = np.sqrt(np.mean(seg * seg)) + 1e-9
        db.append(20.0 * np.log10(rms))
        times.append(start / AUDIO_SR)
    times, db = np.array(times), np.array(db)
    # smooth
    k = max(1, int(SMOOTH_SEC / HOP_SEC))
    if k > 1 and len(db) >= k:
        db = np.convolve(db, np.ones(k) / k, mode="same")
    return times, db

def find_highlights(path):
    """Return a list of (start, end) second-windows of action in the clip."""
    samples = load_audio(path)
    times, db = loudness_curve(samples)
    if len(db) == 0:
        return []

    clip_dur = times[-1] + WIN_SEC

    # Position-dependent loudness bar: a single percentile gives one global
    # threshold; we nudge it by clip position so the replay-buffer payoff (near
    # the end) is easy to clear, while the start has to be genuinely loud.
    base = max(np.percentile(db, LOUDNESS_PERCENTILE), ABS_FLOOR_DB)
    thr = np.full_like(db, base)
    thr[times >= clip_dur - END_ZONE_SEC] -= END_BOOST_DB
    thr[times <= START_GUARD_SEC]         += START_GUARD_DB
    thr = np.maximum(thr, ABS_FLOOR_DB)
    mask = db >= thr

    # contiguous runs of "loud" -> raw segments
    segs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            segs.append([times[i], times[j - 1]])
            i = j
        else:
            i += 1

    # pad, clamp
    segs = [[max(0.0, s - PAD_BEFORE), min(clip_dur, e + PAD_AFTER)]
            for s, e in segs]

    # merge ones that are close together
    merged = []
    for s, e in segs:
        if merged and s <= merged[-1][1] + MERGE_GAP:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # length filters
    out = []
    for s, e in merged:
        if e - s < MIN_HL_SEC:
            continue
        if e - s > MAX_HL_SEC:
            e = s + MAX_HL_SEC
        out.append((round(s, 3), round(e, 3)))

    # Safety net: if nothing cleared the bar, keep the single loudest moment so
    # a clip with real action never silently yields zero highlights.
    if not out:
        peak = int(np.argmax(db))
        s = max(0.0, times[peak] - PAD_BEFORE)
        e = min(clip_dur, times[peak] + PAD_AFTER)
        if e - s >= MIN_HL_SEC:
            out = [(round(s, 3), round(e, 3))]

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
        hls = find_highlights(clip)
        kept = [(s, e) for s, e in hls if is_gameplay(hit_times, s, e)]
        skipped = len(hls) - len(kept)
        msg = f"{base}: class={cls} (conf {score:.2f}), {len(kept)} highlight(s)"
        if skipped:
            msg += f", {skipped} skipped (menu/market)"
        print(msg)

        for idx, (s, e) in enumerate(kept, 1):
            out_name = f"{cls}__{base}__hl{idx:02d}.mp4"
            out_path = os.path.join(args.out, out_name)
            cut(clip, s, e, out_path)
            all_outputs.append(out_path)
            print(f"    -> {out_name}  [{s:.1f}s - {e:.1f}s]")

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
