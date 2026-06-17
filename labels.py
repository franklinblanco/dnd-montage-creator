#!/usr/bin/env python3
"""
labels.py — the label store.

Every judge verdict (and every human correction) is written here as a training
example for the local student. One JSON file per clip under .labels/, same caching
convention as .transcripts/. Records are keyed by window; a higher-trust source
(human > claude > local_vlm > trained) overrides a lower one for the same window,
so re-running the seed or correcting in the report never loses the better label.
"""

import json
import os

LABELS_DIR = ".labels"
SOURCE_TRUST = {"human": 3, "claude": 2, "local_vlm": 1, "trained": 0}


def _path(clip):
    return os.path.join(LABELS_DIR, os.path.basename(clip) + ".json")


def load(clip):
    p = _path(clip)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return {"clip": os.path.basename(clip), "records": []}


def _key(window):
    return (round(float(window[0]), 1), round(float(window[1]), 1))


def save_verdict(clip, window, frames_used, verdict, source):
    """Upsert one verdict for a window, keeping the highest-trust source.

    `verdict` is a plain dict (Verdict.to_dict()). Returns the updated store dict.
    """
    os.makedirs(LABELS_DIR, exist_ok=True)
    data = load(clip)
    key = _key(window)
    new_trust = SOURCE_TRUST.get(source, 0)

    kept = []
    for r in data["records"]:
        if _key(r["window"]) == key:
            if SOURCE_TRUST.get(r["source"], 0) > new_trust:
                return data            # existing label is more trusted — leave it
            continue                   # otherwise drop it; the new one replaces it
        kept.append(r)

    kept.append({
        "window": [round(float(window[0]), 2), round(float(window[1]), 2)],
        "frames_used": frames_used,
        "source": source,
        "verdict": verdict,
    })
    data["records"] = kept
    with open(_path(clip), "w") as fh:
        json.dump(data, fh, indent=2)
    return data
