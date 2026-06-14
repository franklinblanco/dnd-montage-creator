#!/usr/bin/env python3
"""
dnd_montage.py — Dark and Darker montage helper

Two jobs:
  1) TIER 1  -> find the loud/action moments in each clip (audio loudness).
  2) TIER 2  -> read which class you were playing from the bottom-left icon.

It does NOT auto-stitch by default. It cuts each highlight into its own file,
named with the detected class, so you can drag them into DaVinci Resolve and
arrange them yourself. (Pass --stitch to also get one concatenated file.)

----------------------------------------------------------------------
SETUP (once)
----------------------------------------------------------------------
  - Install FFmpeg and make sure `ffmpeg` and `ffprobe` are on your PATH.
  - pip install numpy opencv-python

----------------------------------------------------------------------
WORKFLOW
----------------------------------------------------------------------
  Step 1 — find where the class icon sits on screen (do this once):
      python dnd_montage.py calibrate "clips/some_clip.mp4"
      -> writes _calib_frame.png (full frame with a red ROI box drawn)
         and _calib_crop.png (just what's inside the box).
         Open _calib_frame.png, check the red box sits over the class icon.
         If not, tweak ICON_ROI below and run calibrate again.

  Step 2 — build your icon library (once per class you play):
      python dnd_montage.py addtemplate "clips/a_ranger_clip.mp4" --name ranger
      python dnd_montage.py addtemplate "clips/a_rogue_clip.mp4"  --name rogue
      ... repeat for each class. Saves templates/<name>.png

  Step 3 — process a whole folder:
      python dnd_montage.py run --in clips --out output
      -> output/ranger__a_ranger_clip__hl01.mp4, etc.
         Add --stitch to also get output/_montage_all.mp4

Tuning: everything you'll want to touch lives in the CONFIG block below.
The two knobs that matter most are LOUDNESS_PERCENTILE (higher = pickier)
and PAD_BEFORE / PAD_AFTER (how much lead-in / tail each highlight gets).
"""

import argparse
import os
import subprocess
import sys
import glob

import numpy as np
import cv2

# ======================================================================
# CONFIG — tune these
# ======================================================================

# --- Audio / highlight detection (Tier 1) ---
AUDIO_SR          = 8000   # Hz to downsample audio to for analysis (plenty for loudness)
WIN_SEC           = 0.40   # length of each loudness measurement window
HOP_SEC           = 0.10   # how far the window slides each step
SMOOTH_SEC        = 0.50   # moving-average smoothing over the loudness curve

LOUDNESS_PERCENTILE = 90.0 # a moment counts as "action" if louder than this %
                           #   of the clip. Higher = fewer, punchier highlights.
ABS_FLOOR_DB      = -40.0  # never flag anything quieter than this (kills silence)

PAD_BEFORE        = 3.0    # seconds of lead-in before a detected peak
PAD_AFTER         = 5.0    # seconds of tail after it
MERGE_GAP         = 4.0    # merge two highlights if they're within this many sec
MIN_HL_SEC        = 2.0    # drop highlights shorter than this
MAX_HL_SEC        = 30.0   # hard cap on a single highlight's length

# --- Class icon detection (Tier 2) ---
# ROI given as fractions of the frame (x0, y0, x1, y1), origin top-left.
# Default: a box in the bottom-left corner. Tune with `calibrate`.
ICON_ROI          = (0.010, 0.880, 0.090, 0.985)
ICON_SAMPLES      = 3      # how many frames to sample per clip, then majority-vote
MATCH_THRESHOLD   = 0.60   # min normalized-correlation score to accept a class
SEARCH_PAD_PX     = 12     # let the icon wiggle this many px when matching

# --- Output encoding ---
# Re-encoding gives frame-accurate cuts (stream-copy only cuts on keyframes).
VIDEO_CODEC       = "libx264"
CRF               = "18"
PRESET            = "veryfast"
AUDIO_CODEC       = "aac"

TEMPLATE_DIR      = "templates"

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

    thresh = max(np.percentile(db, LOUDNESS_PERCENTILE), ABS_FLOOR_DB)
    mask = db >= thresh

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

    duration = times[-1] + WIN_SEC
    # pad, clamp
    segs = [[max(0.0, s - PAD_BEFORE), min(duration, e + PAD_AFTER)]
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
    return out

# ======================================================================
# TIER 2 — class icon detection (template matching)
# ======================================================================

def roi_pixels(w, h):
    x0, y0, x1, y1 = ICON_ROI
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)

def crop_roi(img):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi_pixels(w, h)
    return img[y0:y1, x0:x1]

def detect_class(path, duration, templates):
    """Sample frames, match the icon ROI against each template, majority-vote."""
    if not templates:
        return None, 0.0

    # evenly spaced sample times, avoiding the very edges
    ts = np.linspace(0.15 * duration, 0.85 * duration, ICON_SAMPLES)
    votes = {}
    best_overall = 0.0
    tmp = "_cls_frame.png"

    for t in ts:
        grab_frame(path, float(t), tmp)
        frame = cv2.imread(tmp)
        if frame is None:
            continue
        region = crop_roi(frame)
        rh, rw = region.shape[:2]

        best_name, best_score = None, 0.0
        for name, tpl in templates.items():
            th, tw = tpl.shape[:2]
            # template must fit inside the search region
            if th > rh or tw > rw:
                tpl_fit = cv2.resize(tpl, (min(tw, rw), min(th, rh)))
            else:
                tpl_fit = tpl
            res = cv2.matchTemplate(region, tpl_fit, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score > best_score:
                best_name, best_score = name, score

        if best_name and best_score >= MATCH_THRESHOLD:
            votes[best_name] = votes.get(best_name, 0) + 1
            best_overall = max(best_overall, best_score)

    if os.path.exists(tmp):
        os.remove(tmp)

    if not votes:
        return None, 0.0
    winner = max(votes, key=votes.get)
    return winner, best_overall

def load_templates():
    templates = {}
    if not os.path.isdir(TEMPLATE_DIR):
        return templates
    for f in glob.glob(os.path.join(TEMPLATE_DIR, "*.png")):
        name = os.path.splitext(os.path.basename(f))[0]
        img = cv2.imread(f)
        if img is not None:
            templates[name] = img
    return templates

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
    x0, y0, x1, y1 = roi_pixels(w, h)
    crop = frame[y0:y1, x0:x1].copy()
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite("_calib_frame.png", frame)
    cv2.imwrite("_calib_crop.png", crop)
    print(f"Frame is {w}x{h}. ROI pixels: x {x0}-{x1}, y {y0}-{y1}")
    print("Open _calib_frame.png — the red box should sit over the class icon.")
    print("If it's off, edit ICON_ROI and run calibrate again.")

def mode_addtemplate(args):
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    dur = probe_duration(args.clip)
    grab_frame(args.clip, dur / 2.0, "_tpl_frame.png")
    frame = cv2.imread("_tpl_frame.png")
    if frame is None:
        sys.exit("Could not read a frame from that clip.")
    crop = crop_roi(frame)
    out = os.path.join(TEMPLATE_DIR, f"{args.name}.png")
    cv2.imwrite(out, crop)
    os.remove("_tpl_frame.png")
    print(f"Saved template for '{args.name}' -> {out}")
    print("Tip: eyeball it to confirm the icon is centered and not cut off.")

def mode_run(args):
    os.makedirs(args.out, exist_ok=True)
    templates = load_templates()
    if not templates:
        print("WARNING: no templates found — clips will be labeled 'unknown'.\n"
              "         Run `addtemplate` first for class detection.\n")

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

        cls, score = detect_class(clip, dur, templates)
        cls = cls or "unknown"
        hls = find_highlights(clip)
        print(f"{base}: class={cls} (conf {score:.2f}), "
              f"{len(hls)} highlight(s)")

        for idx, (s, e) in enumerate(hls, 1):
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

    c = sub.add_parser("calibrate", help="check where the class-icon ROI lands")
    c.add_argument("clip")
    c.set_defaults(func=mode_calibrate)

    a = sub.add_parser("addtemplate", help="save the ROI of a clip as a class template")
    a.add_argument("clip")
    a.add_argument("--name", required=True, help="class name, e.g. ranger")
    a.set_defaults(func=mode_addtemplate)

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
