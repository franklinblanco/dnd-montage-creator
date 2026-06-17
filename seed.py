#!/usr/bin/env python3
"""
seed.py — label candidate windows with a model judge.

Runs the existing high-recall detector to get candidate fight windows, samples
frames from each, and labels them, writing every verdict to the .labels/ store —
the training set for the free local student. Two backends (`--judge`):

  - claude (Phase 1): Opus via the Batches API (−50% cost). The ONLY step that
    costs money; this is the one-time ~$1 seed. Needs ANTHROPIC_API_KEY, no GPU.
  - local (Phase 2): the free Qwen2.5-VL-7B teacher via Ollama on the PC (CUDA).
    Use it to label new clips going forward, and to build the calibration set
    (re-label the seed windows, then compare with calibrate.py).

See DESIGN.md §10.

Usage:
    # one-time paid seed (Mac or PC):
    pip install anthropic            # plus the existing requirements + ffmpeg
    export ANTHROPIC_API_KEY=...     # PowerShell: $env:ANTHROPIC_API_KEY="..."
    python seed.py --in clips --dry-run       # preview windows + cost, no API call
    python seed.py --in clips                 # run the one-time seed (~$1)
    python seed.py --in clips --max-windows 40

    # free local labeling (PC, needs `ollama serve` + `ollama pull qwen2.5vl:7b`):
    python seed.py --in clips --judge local
"""

import argparse
import glob
import os
import sys
import time

import dnd_montage as dm
import labels as label_store
from frames import sample_window
from judge import ClaudeJudge, LocalVLMJudge, SEED_MODEL, WindowContext

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


def _ctx(clip, s, e, dur):
    return WindowContext(clip=os.path.basename(clip), start=s, end=e,
                         duration=dur, voice=voice_in_window(clip, s, e))


def run_local(jobs, args):
    """Label every window synchronously with the free local Qwen VLM (no cost,
    no batch API). Writes source='local_vlm' so the trust order in labels.py keeps
    any existing Claude/human label for the same window."""
    judge = LocalVLMJudge(model=args.model, host=args.ollama_host)
    print(f"Labeling locally with {args.model} via Ollama ({args.ollama_host}) ...")
    ok = err = pvp = 0
    score_sum = 0.0
    for i, (clip, (s, e), dur) in enumerate(jobs):
        name = os.path.basename(clip)
        frames = sample_window(clip, s, e, n=args.frames, width=FRAME_WIDTH)
        if not frames:
            print(f"  ! no frames for {name} [{s:.1f}-{e:.1f}]")
            err += 1
            continue
        try:
            verdict = judge.score(frames, _ctx(clip, s, e, dur))
        except Exception as ex:
            err += 1
            print(f"  ! {name} [{s:.1f}-{e:.1f}] failed: {ex}")
            continue
        label_store.save_verdict(clip, (s, e), len(frames), verdict.to_dict(),
                                 source="local_vlm")
        ok += 1
        pvp += 1 if verdict.is_pvp else 0
        score_sum += verdict.montage_score
        print(f"  [{i + 1}/{len(jobs)}] {name} [{s:.1f}-{e:.1f}] "
              f"pvp={verdict.is_pvp} score={verdict.montage_score} "
              f"conf={verdict.confidence:.2f}")
    print(f"\nDone. {ok} labeled, {err} errored. PvP: {pvp}/{ok or 1}. "
          f"Avg score: {score_sum / max(1, ok):.1f}. Cost: free (local).")
    print(f"Labels written under {label_store.LABELS_DIR}/.")


def run_claude(jobs, args):
    judge = ClaudeJudge(model=args.model, effort=args.effort)

    print("Sampling frames + building batch ...")
    requests, meta = [], {}
    for i, (clip, (s, e), dur) in enumerate(jobs):
        frames = sample_window(clip, s, e, n=args.frames, width=FRAME_WIDTH)
        if not frames:
            print(f"  ! no frames for {os.path.basename(clip)} [{s:.1f}-{e:.1f}]")
            continue
        cid = f"w{i:04d}"
        requests.append(judge.build_request(cid, frames, _ctx(clip, s, e, dur)))
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


def main():
    ap = argparse.ArgumentParser(
        description="Label candidate windows with a model judge (Phase 1 seed / Phase 2 local)")
    ap.add_argument("--in", dest="in_dir", default="clips")
    ap.add_argument("--judge", choices=["claude", "local"], default="claude",
                    help="claude = one-time paid Opus seed; local = free Qwen via Ollama")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="cap the number of windows labeled (bounds spend on claude)")
    ap.add_argument("--model", default=None,
                    help="override the model id (defaults per --judge)")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"],
                    help="claude only: reasoning effort")
    ap.add_argument("--ollama-host", default="http://localhost:11434",
                    help="local only: Ollama base URL")
    ap.add_argument("--frames", type=int, default=FRAMES_PER_WINDOW)
    ap.add_argument("--dry-run", action="store_true",
                    help="list windows (+ cost for claude), sample/call nothing")
    args = ap.parse_args()
    if args.model is None:
        args.model = SEED_MODEL if args.judge == "claude" else "qwen2.5vl:7b"

    clips = find_clips(args.in_dir)
    if not clips:
        sys.exit(f"No video files in {args.in_dir} (labeling needs the SOURCE clips).")

    jobs = collect_windows(clips, args.max_windows)
    if not jobs:
        sys.exit("No candidate windows found.")
    print(f"{len(clips)} clip(s), {len(jobs)} candidate window(s). Judge: {args.judge}.")
    if args.judge == "claude":
        print(f"Estimated seed cost: ~${len(jobs) * EST_PER_WINDOW:.2f} "
              f"(Opus via Batches, ~${EST_PER_WINDOW:.4f}/window).")
    else:
        print("Local judge — free (no API cost).")

    if args.dry_run:
        for clip, (s, e), _dur in jobs:
            print(f"  {os.path.basename(clip)}  [{s:.1f}-{e:.1f}]  ({e - s:.0f}s)")
        print("\nDry run — no frames sampled, no model call.")
        return

    (run_claude if args.judge == "claude" else run_local)(jobs, args)


if __name__ == "__main__":
    main()
