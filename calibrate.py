#!/usr/bin/env python3
"""
calibrate.py — measure how much to trust the free local teacher (Phase 2 step 2).

Re-labels the windows the paid Opus seed already labeled (source="claude" in
.labels/) with the local Qwen VLM, then reports agreement against the seed:
is_pvp accuracy + the PvP confusion matrix (treating Claude as ground truth),
montage_score MAE and within-1 accuracy, and mean category overlap. Run this once
after the seed so you know the local teacher's error rate before using it at scale
(DESIGN.md §6/§10, HANDOFF.md Phase 2 step 2).

The local verdicts are NOT written to .labels/ — the store keeps the higher-trust
Claude label for each seeded window, so a local re-label would be dropped anyway.
Pass --out to dump the per-window comparison for inspection.

Usage (PC, with `ollama serve` running + `ollama pull qwen2.5vl:7b`):
    python calibrate.py --in clips
    python calibrate.py --in clips --out reports/calibration.json
"""

import argparse
import glob
import json
import os
import sys

import labels as label_store
from frames import sample_window
from judge import CATEGORIES, LocalVLMJudge, Verdict, WindowContext

FRAME_WIDTH = 512
DEFAULT_FRAMES = 9


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _seeded_records():
    """[(clip_basename, record)] for every claude-sourced window in the store."""
    out = []
    for p in sorted(glob.glob(os.path.join(label_store.LABELS_DIR, "*.json"))):
        with open(p) as fh:
            data = json.load(fh)
        for r in data.get("records", []):
            if r.get("source") == "claude":
                out.append((data["clip"], r))
    return out


def _context(clip_path, s, e):
    """Best-effort match to the seed's context (duration + overlapping voice).
    dnd_montage is heavy (numpy/cv2/whisper); import lazily so calibrate stays
    usable even if a window's context can't be rebuilt."""
    dur, voice = 0.0, ""
    try:
        import dnd_montage as dm
        dur = dm.probe_duration(clip_path)
        try:
            segs = dm.transcribe(clip_path)
            voice = " ".join(t for (a, b, t) in segs if b >= s and a <= e)[:500]
        except Exception:
            pass
    except Exception:
        pass
    return WindowContext(clip=os.path.basename(clip_path), start=s, end=e,
                         duration=dur, voice=voice)


def main():
    ap = argparse.ArgumentParser(description="Calibrate the local VLM against the Claude seed")
    ap.add_argument("--in", dest="in_dir", default="clips",
                    help="folder with the SOURCE clips (to re-sample frames)")
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="cap windows compared (quick spot-check)")
    ap.add_argument("--out", default=None, help="write per-window comparison JSON here")
    args = ap.parse_args()

    seeded = _seeded_records()
    if not seeded:
        sys.exit("No claude-sourced labels in .labels/ — run the seed first "
                 "(python seed.py --in clips).")
    if args.max_windows:
        seeded = seeded[:args.max_windows]

    judge = LocalVLMJudge(model=args.model, host=args.ollama_host)
    print(f"Calibrating {args.model} against {len(seeded)} Claude-seeded window(s) ...")

    rows = []
    tp = fp = fn = tn = 0          # local vs claude-as-truth, on is_pvp
    abs_err = exact = within1 = 0.0
    jac_sum = 0.0
    missing = skipped = 0

    for i, (clip_name, rec) in enumerate(seeded):
        clip_path = os.path.join(args.in_dir, clip_name)
        if not os.path.exists(clip_path):
            missing += 1
            print(f"  ? clip not in --in: {clip_name} (skipped)")
            continue
        s, e = rec["window"]
        frames = sample_window(clip_path, s, e, n=args.frames, width=FRAME_WIDTH)
        if not frames:
            skipped += 1
            print(f"  ! no frames for {clip_name} [{s:.1f}-{e:.1f}]")
            continue
        try:
            local = judge.score(frames, _context(clip_path, s, e))
        except Exception as ex:
            skipped += 1
            print(f"  ! {clip_name} [{s:.1f}-{e:.1f}] local judge failed: {ex}")
            continue

        ref = Verdict.from_dict(rec["verdict"])
        # PvP confusion (Claude = ground truth)
        if ref.is_pvp and local.is_pvp:
            tp += 1
        elif not ref.is_pvp and local.is_pvp:
            fp += 1
        elif ref.is_pvp and not local.is_pvp:
            fn += 1
        else:
            tn += 1
        d = abs(local.montage_score - ref.montage_score)
        abs_err += d
        exact += 1 if d == 0 else 0
        within1 += 1 if d <= 1 else 0
        jac = _jaccard(local.categories, ref.categories)
        jac_sum += jac

        rows.append({
            "clip": clip_name, "window": [s, e],
            "claude": {"is_pvp": ref.is_pvp, "score": ref.montage_score,
                       "categories": ref.categories},
            "local": {"is_pvp": local.is_pvp, "score": local.montage_score,
                      "categories": local.categories, "confidence": local.confidence},
            "score_abs_err": d, "category_jaccard": round(jac, 3),
        })
        agree = "OK" if ref.is_pvp == local.is_pvp else "XX"
        print(f"  [{i + 1}/{len(seeded)}] {agree} {clip_name} [{s:.1f}-{e:.1f}] "
              f"pvp c={ref.is_pvp}/l={local.is_pvp} score c={ref.montage_score}/l={local.montage_score}")

    n = tp + fp + fn + tn
    if not n:
        sys.exit("Nothing compared (no clips matched --in, or all windows failed).")

    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec_ = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec_ / (prec + rec_)
          if (tp + fp) and (tp + fn) and (prec + rec_) else float("nan"))

    print("\n=== Calibration vs Claude seed ===")
    print(f"windows compared : {n}  (missing clips: {missing}, frame/judge fails: {skipped})")
    print(f"is_pvp agreement : {acc:.1%}")
    print(f"  confusion (Claude=truth): TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  local PvP precision={prec:.2f} recall={rec_:.2f} f1={f1:.2f}")
    print(f"montage_score    : MAE={abs_err / n:.2f}  exact={exact / n:.1%}  within±1={within1 / n:.1%}")
    print(f"category overlap : mean Jaccard={jac_sum / n:.2f}")
    print("\nTrust guide: high is_pvp agreement + low score MAE -> the free teacher can "
          "label new clips at scale. Otherwise keep using the seed / human review on the gap.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        summary = {
            "model": args.model, "windows": n,
            "missing_clips": missing, "failed": skipped,
            "is_pvp_accuracy": acc, "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "pvp_precision": prec, "pvp_recall": rec_, "pvp_f1": f1,
            "score_mae": abs_err / n, "score_exact": exact / n, "score_within1": within1 / n,
            "category_jaccard_mean": jac_sum / n,
            "categories": CATEGORIES, "rows": rows,
        }
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nPer-window comparison written to {args.out}")


if __name__ == "__main__":
    main()
