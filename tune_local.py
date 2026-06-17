#!/usr/bin/env python3
"""
tune_local.py — experiment harness to improve the local VLM's PvP call.

Scores every human-labeled window with several (prompt, frames, schema) variants
and reports each variant's agreement vs the human ground truth. In-memory only —
never writes .labels/. Throwaway tuning tool; the winning config gets baked into
judge.py.
"""
import base64, json, os, time, urllib.request
import glob

import dnd_montage as dm
import labels as label_store
from frames import sample_window
from judge import output_schema, _parse_verdict, _window_header, Verdict, WindowContext

HOST = "http://localhost:11434"
MODEL = "qwen2.5vl:7b"
CLIPS = r"C:\Users\Frank\Videos"

# ---- prompts --------------------------------------------------------------

SYS_BASE = """\
You label short windows of Dark and Darker gameplay, recorded from the player's own
point of view, to find montage-worthy PvP moments.

Dark and Darker has NO kill feed and NO enemy nameplates, so judge from what you see
across the frames. They are in time order, each tagged with its [t=..s] timestamp.

PvP vs PvE — the key call:
- PvP = another human PLAYER is involved. Players wear varied, mismatched craftable
  armor and weapons, use class abilities and consumables, and move erratically
  (strafing, jukes, crouch-spam, kiting). Health and stamina swing fast.
- PvE = only AI monsters (skeletons, zombies, goblins, cultists, spiders, ghosts,
  wraiths, etc.) and traps. Mobs look uniform and move in repetitive, predictable
  patterns.
- The recording player's OWN abilities, arrows, spells, and pets are not an
  opponent. If you only see mobs or an empty room, is_pvp is false.

categories (choose any that apply; [] if none):
- kill_win: the player kills an enemy player or clearly wins the fight.
- clutch_lowhp: outnumbered (1vX), very low HP, or a comeback.
- funny_fail: death to a trap, a big whiff, trolling, or otherwise funny/meme.
- flashy_play: a big hit, clean shot, or slick ability combo, even without a kill.

montage_score (0-10): 0-2 no PvP/boring; 3-5 minor PvP; 6-8 solid PvP; 9-10 exceptional.

Also return tight_start/tight_end (sec, clip-relative), confidence (0-1), one-line reason.
Be decisive; when ambiguous, lower the confidence rather than inflating the score."""

SYS_V1 = """\
You label short windows of Dark and Darker gameplay (the player's own point of view)
to find montage-worthy PvP moments. These clips were saved by a REPLAY BUFFER right
after the player got into a fight, so most windows DO contain a real player-vs-player
fight somewhere. Your main job is to tell PvP apart from PvE.

FIRST decide is_pvp: is another human PLAYER involved, or only AI monsters/traps?
Dark and Darker has NO kill feed and NO nameplates, so infer from these PvP cues —
if you see ANY of them, it is PvP:
- An opponent in varied, mismatched CRAFTED armor or distinctive weapons (not a
  uniform monster model).
- The opponent uses class abilities, spells, thrown consumables, or potions.
- Erratic HUMAN movement: strafing, juking, crouch-spam, jumping, kiting, retreating,
  chasing.
- Fast back-and-forth health/stamina swings, or the player reacting to an unseen
  ranged threat (dodging arrows/spells).
PvE = only skeletons, zombies, goblins, cultists, spiders, ghosts, wraiths, etc.:
uniform models, repetitive predictable patterns, no class abilities. The recording
player's OWN arrows/spells/pets are not the opponent — look for the separate enemy.

THEN score montage-worthiness 0-10 — but a window that is NOT PvP scores at most 2.
Never give PvE combat a high score.
- 0-2: no PvP / boring / looting / only mobs.
- 3-5: minor PvP poke or an unremarkable fight.
- 6-8: a solid PvP kill or a good back-and-forth fight.
- 9-10: exceptional — multi-kill, clutch 1vX, or genuinely hilarious.

categories (any that apply; [] if none): kill_win (kills a player / wins),
clutch_lowhp (1vX, very low HP, comeback), funny_fail (trap death, big whiff,
troll/meme), flashy_play (big hit, clean shot, slick combo).

Also return tight_start/tight_end (sec, clip-relative) trimmed to the action,
confidence 0-1, and a one-line reason. Make a decisive PvP call from the visual cues."""

# reason-first schema: an analysis field generated BEFORE is_pvp (constrained
# decoding emits properties in order -> the model "thinks" before deciding).
def schema_reasoned():
    s = output_schema()
    props = {"visual_analysis": {"type": "string"}}
    props.update(s["properties"])
    s["properties"] = props
    s["required"] = ["visual_analysis"] + s["required"]
    return s


def score(system, frames, ctx, schema, num_ctx):
    ts = ", ".join(f"{fr.t:.1f}s" for fr in frames)
    prompt = (f"{_window_header(frames, ctx)}\nThe frames are in time order, at: {ts}.\n"
              "Return the JSON verdict for this window.")
    payload = {
        "model": MODEL, "stream": False, "format": schema,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt,
             "images": [base64.standard_b64encode(f.jpeg).decode() for f in frames]},
        ],
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }
    req = urllib.request.Request(HOST + "/api/chat",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    body = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
    return _parse_verdict(body["message"]["content"]), body.get("prompt_eval_count", 0)


def human_windows():
    out = []
    for p in sorted(glob.glob(os.path.join(label_store.LABELS_DIR, "*.json"))):
        d = json.load(open(p))
        for r in d["records"]:
            if r["source"] == "human":
                out.append((d["clip"], r["window"], Verdict.from_dict(r["verdict"])))
    return out


def ctx_for(clip, s, e):
    path = os.path.join(CLIPS, clip)
    dur = dm.probe_duration(path)
    voice = " ".join(t for (a, b, t) in dm.transcribe(path) if b >= s and a <= e)[:300]
    return path, WindowContext(clip=clip, start=s, end=e, duration=dur, voice=voice)


VARIANTS = [
    {"name": "V0 base 9@512",     "sys": SYS_BASE, "n": 9,  "w": 512, "schema": output_schema(),  "ctx": 16384},
    {"name": "V1 prompt 9@512",   "sys": SYS_V1,   "n": 9,  "w": 512, "schema": output_schema(),  "ctx": 16384},
    {"name": "V2 +reason 9@512",  "sys": SYS_V1,   "n": 9,  "w": 512, "schema": schema_reasoned(), "ctx": 16384},
    {"name": "V3 +frames 12@640", "sys": SYS_V1,   "n": 12, "w": 640, "schema": output_schema(),  "ctx": 32768},
]


def main():
    wins = human_windows()
    print(f"{len(wins)} human-labeled windows.\n")
    # pre-resolve contexts + frame sets per (n,w) to avoid recutting per variant
    summary = []
    for V in VARIANTS:
        tp = fp = fn = tn = 0
        abs_err = 0.0
        toks = 0
        t0 = time.time()
        print(f"=== {V['name']} ===")
        for clip, (s, e), truth in wins:
            path, ctx = ctx_for(clip, s, e)
            frames = sample_window(path, s, e, n=V["n"], width=V["w"])
            try:
                v, tk = score(V["sys"], frames, ctx, V["schema"], V["ctx"])
            except Exception as ex:
                print(f"  ! {clip[:30]} [{s:.0f}-{e:.0f}] fail: {str(ex)[:120]}")
                continue
            toks = max(toks, tk)
            if truth.is_pvp and v.is_pvp: tp += 1
            elif not truth.is_pvp and v.is_pvp: fp += 1
            elif truth.is_pvp and not v.is_pvp: fn += 1
            else: tn += 1
            abs_err += abs(v.montage_score - truth.montage_score)
            flag = "OK" if truth.is_pvp == v.is_pvp else "XX"
            print(f"  {flag} {clip[:34]:<34} pvp r={truth.is_pvp!s:<5}/l={v.is_pvp!s:<5} "
                  f"score r={truth.montage_score}/l={v.montage_score}")
        n = tp + fp + fn + tn
        acc = (tp + tn) / n if n else 0
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if (tp + fp) and (tp + fn) and (prec + rec) else float("nan")
        mae = abs_err / n if n else 0
        print(f"  -> acc={acc:.1%} P={prec:.2f} R={rec:.2f} F1={f1:.2f} MAE={mae:.2f} "
              f"(TP{tp} FP{fp} FN{fn} TN{tn}) ~{toks}tok {time.time()-t0:.0f}s\n")
        summary.append((V["name"], acc, prec, rec, f1, mae, tp, fp, fn, tn))

    print("\n================ SUMMARY (vs human, N=%d) ================" % len(wins))
    print(f"{'variant':<20}{'acc':>7}{'prec':>7}{'rec':>7}{'F1':>7}{'MAE':>7}  confusion")
    for name, acc, prec, rec, f1, mae, tp, fp, fn, tn in summary:
        print(f"{name:<20}{acc:>6.0%}{prec:>7.2f}{rec:>7.2f}{f1:>7.2f}{mae:>7.2f}  "
              f"TP{tp} FP{fp} FN{fn} TN{tn}")


if __name__ == "__main__":
    main()
