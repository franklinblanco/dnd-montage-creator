#!/usr/bin/env python3
"""
seed.py — the one-time paid seed (Phase 1).

Runs the existing high-recall detector to get candidate fight windows, samples
frames from each, and labels them with Opus (Claude) via the Batches API (−50%
cost). Every verdict is written to the .labels/ store — the training set for the
free local student. This is the ONLY step that costs money; everything after is
local. See DESIGN.md §10.

This step needs no GPU — it runs wherever the source clips + existing deps live
(e.g. the Mac). The local VLM and student training (later phases) use the PC.

Usage:
    pip install anthropic            # plus the existing requirements + ffmpeg
    export ANTHROPIC_API_KEY=...     # needed for THIS step only
    python seed.py --in clips --dry-run       # preview windows + cost, no API call
    python seed.py --in clips                 # run the one-time seed (~$1)
    python seed.py --in clips --max-windows 40
"""

import argparse
import glob
import os
import sys
import time

import dnd_montage as dm
import labels as label_store
from frames import sample_window
from judge import ClaudeJudge, SEED_MODEL, WindowContext

FRAMES_PER_WINDOW = 9
FRAME_WIDTH = 512
# Opus 4.8 via Batches (−50%): ~$2.5/1M input, ~$12.5/1M output. A 512px frame is
# ~200 image tokens; ~9 frames + prompt ≈ 2k input, small JSON out → ~$0.0075/window.
EST_PER_WINDOW = 0.0075


def find_clips(in_dir):
    clips = []
    for ext in ("mp4", "mkv", "mov", "MP4", "MKV", "MOV"):
        clips += glob.glob(os.path.join(in_dir, f"*.{ext}"))
    return sorted(set(clips))


def voice_in_window(clip, s, e):
    """Narration overlapping [s, e], from the cached Whisper transcript."""
    try:
        segs = dm.transcribe(clip)
    except Exception:
        return ""
    parts = [t for (a, b, t) in segs if b >= s and a <= e]
    return " ".join(parts)[:500]


def collect_windows(clips, max_windows=None):
    """[(clip, (start, end), duration)] for every candidate window in the folder."""
    jobs = []
    for clip in clips:
        try:
            dur = dm.probe_duration(clip)
        except RuntimeError as ex:
            print(f"  ! skip {os.path.basename(clip)}: {ex}")
            continue
        for (s, e, _w, _kill) in dm.fight_windows(clip, dur):
            jobs.append((clip, (s, e), dur))
    return jobs[:max_windows] if max_windows else jobs


def main():
    ap = argparse.ArgumentParser(description="One-time Claude seed labeling (Phase 1)")
    ap.add_argument("--in", dest="in_dir", default="clips")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="cap the number of windows sent (bounds spend)")
    ap.add_argument("--model", default=SEED_MODEL)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--frames", type=int, default=FRAMES_PER_WINDOW)
    ap.add_argument("--dry-run", action="store_true",
                    help="list windows + cost estimate, sample/call nothing")
    args = ap.parse_args()

    clips = find_clips(args.in_dir)
    if not clips:
        sys.exit(f"No video files in {args.in_dir} (the seed needs the SOURCE clips).")

    jobs = collect_windows(clips, args.max_windows)
    if not jobs:
        sys.exit("No candidate windows found.")
    print(f"{len(clips)} clip(s), {len(jobs)} candidate window(s).")
    print(f"Estimated seed cost: ~${len(jobs) * EST_PER_WINDOW:.2f} "
          f"(Opus via Batches, ~${EST_PER_WINDOW:.4f}/window).")

    if args.dry_run:
        for clip, (s, e), _dur in jobs:
            print(f"  {os.path.basename(clip)}  [{s:.1f}-{e:.1f}]  ({e - s:.0f}s)")
        print("\nDry run — no frames sampled, no API call.")
        return

    judge = ClaudeJudge(model=args.model, effort=args.effort)

    print("Sampling frames + building batch ...")
    requests, meta = [], {}
    for i, (clip, (s, e), dur) in enumerate(jobs):
        frames = sample_window(clip, s, e, n=args.frames, width=FRAME_WIDTH)
        if not frames:
            print(f"  ! no frames for {os.path.basename(clip)} [{s:.1f}-{e:.1f}]")
            continue
        ctx = WindowContext(clip=os.path.basename(clip), start=s, end=e,
                            duration=dur, voice=voice_in_window(clip, s, e))
        cid = f"w{i:04d}"
        requests.append(judge.build_request(cid, frames, ctx))
        meta[cid] = (clip, (s, e), len(frames))

    if not requests:
        sys.exit("No frames could be sampled (are the source clips present?).")

    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    print(f"Submitted batch {batch.id} ({len(requests)} requests). Polling ...")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        rc = b.request_counts
        print(f"  status={b.processing_status} done={rc.succeeded + rc.errored}/{len(requests)}")
        time.sleep(30)

    ok = err = pvp = 0
    score_sum = 0.0
    in_full = cache_r = cache_w = out_tok = 0
    for res in client.messages.batches.results(batch.id):
        clip, window, nframes = meta[res.custom_id]
        if res.result.type != "succeeded":
            err += 1
            print(f"  ! {res.custom_id} {os.path.basename(clip)}: {res.result.type}")
            continue
        msg = res.result.message
        u = msg.usage
        in_full += u.input_tokens or 0
        cache_r += getattr(u, "cache_read_input_tokens", 0) or 0
        cache_w += getattr(u, "cache_creation_input_tokens", 0) or 0
        out_tok += u.output_tokens or 0
        try:
            verdict = ClaudeJudge.parse_message(msg)
        except Exception as ex:
            err += 1
            print(f"  ! {res.custom_id} parse failed: {ex}")
            continue
        label_store.save_verdict(clip, window, nframes, verdict.to_dict(), source="claude")
        ok += 1
        pvp += 1 if verdict.is_pvp else 0
        score_sum += verdict.montage_score

    # Actual cost at Opus batch rates (input 2.5, cache-write 3.125, cache-read 0.25,
    # output 12.5 per 1M tokens).
    cost = (in_full / 1e6 * 2.5 + cache_w / 1e6 * 3.125
            + cache_r / 1e6 * 0.25 + out_tok / 1e6 * 12.5)
    print(f"\nDone. {ok} labeled, {err} errored. PvP: {pvp}/{ok or 1}. "
          f"Avg score: {score_sum / max(1, ok):.1f}.")
    print(f"Tokens: in={in_full} cache_w={cache_w} cache_r={cache_r} out={out_tok}. "
          f"Actual cost ~= ${cost:.2f}.")
    print(f"Labels written under {label_store.LABELS_DIR}/.")


if __name__ == "__main__":
    main()
