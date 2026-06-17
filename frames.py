#!/usr/bin/env python3
"""
frames.py — sample frames from a clip window for the model judge.

The judge (VLM or trained head) never watches a whole clip — it looks at a handful
of frames from a candidate fight window. We sample N frames evenly across
[start, end], downscaled, as JPEG bytes ready to base64 into an API request or feed
a local model. Shared by every judge backend (Claude seed, local VLM, trainer).
"""

import glob
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class Frame:
    t: float      # timestamp in seconds, clip-relative
    jpeg: bytes   # downscaled JPEG bytes


def sample_window(path, start, end, n=9, width=512):
    """Return up to ~n Frames evenly spaced across [start, end], scaled to `width`
    px wide (height auto, kept even). One ffmpeg call per window.

    Uses `-ss <start> -i ... -t <span>` (seek then bounded duration) — unambiguous
    across ffmpeg versions, unlike `-ss ... -to` whose origin is version-dependent.
    """
    start = max(0.0, float(start))
    span = max(0.1, float(end) - start)
    fps = n / span
    tmp = tempfile.mkdtemp(prefix="dndframes_")
    try:
        subprocess.run(
            ["ffmpeg", "-ss", f"{start:.3f}", "-i", path, "-t", f"{span:.3f}",
             "-vf", f"fps={fps:.6f},scale={width}:-2", "-q:v", "4",
             os.path.join(tmp, "f_%03d.jpg"), "-y", "-v", "quiet"],
            capture_output=True,
        )
        files = sorted(glob.glob(os.path.join(tmp, "f_*.jpg")))
        frames = []
        count = len(files)
        for i, fp in enumerate(files):
            # place each sampled frame at the center of its time bin
            t = start + (i + 0.5) * span / max(1, count)
            with open(fp, "rb") as fh:
                frames.append(Frame(round(t, 2), fh.read()))
        return frames
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
