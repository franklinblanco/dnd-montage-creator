#!/usr/bin/env python3
"""
workbench.py — fast clip workbench to replace the manual DaVinci grind.

Pick a clip from the bar; it loads the FULL clip with an audio WAVEFORM under it
so you can see the action spikes (gunshots/fights are loud). Drag a region on the
waveform (or Set IN/OUT from the playhead) and Add cut — as many cuts per clip as
you want. All cuts collect into one montage list you can drag to reorder. Then
export three ways:

  A) folder of frame-accurate trimmed MP4s, named in chronological order.
  B) DaVinci timeline (FCPXML primary + EDL fallback) referencing the cuts.
  C) full auto-montage (cuts stitched, + optional intro/outro/music).

The detector's candidate windows are drawn on the waveform as suggestions, and
the project (your cuts) persists to .workbench/project.json.

Usage:
    python workbench.py --in "C:\\Users\\Frank\\Videos"
    # browser opens at http://localhost:8000 ; exports land under montage/
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

import labels as label_store
from judge import CATEGORIES

WB_DIR = ".workbench"
FULL_DIR = os.path.join(WB_DIR, "full")        # remuxed full-clip mp4s (browser-playable)
WAVE_DIR = os.path.join(WB_DIR, "wave")        # cached waveform peak JSON
PROJECT = os.path.join(WB_DIR, "project.json")
KEEP_SCORE = 6                                  # seed cuts from is_pvp windows scoring >= this
WAVE_BUCKETS = 1600
VCODEC, CRF, PRESET, ACODEC = "libx264", "18", "veryfast", "aac"
TS_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})[ _-](\d{2})-(\d{2})-(\d{2})")


# ---- helpers --------------------------------------------------------------

def _safe(name, n=80):
    out = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return out[:n].strip("_") or "clip"


def recording_key(clip, clip_path):
    m = TS_RE.search(clip)
    if m:
        return time.mktime((int(m[1]), int(m[2]), int(m[3]),
                            int(m[4]), int(m[5]), int(m[6]), 0, 0, -1))
    try:
        return os.path.getmtime(clip_path)
    except OSError:
        return 0.0


def find_clips(in_dir):
    clips = []
    for ext in ("mp4", "mkv", "mov", "MP4", "MKV", "MOV"):
        clips += glob.glob(os.path.join(in_dir, f"*.{ext}"))
    return sorted(set(clips))


def _duration(path):
    try:
        return float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path], capture_output=True, text=True).stdout or 0)
    except Exception:
        return 0.0


def windows_from_labels():
    out = []
    for p in sorted(glob.glob(os.path.join(label_store.LABELS_DIR, "*.json"))):
        data = json.load(open(p))
        best = {}
        for r in data["records"]:
            key = (round(r["window"][0], 1), round(r["window"][1], 1))
            if key not in best or r["source"] == "human":
                best[key] = r
        for r in best.values():
            s, e = r["window"]
            out.append((data["clip"], s, e, r["verdict"]))
    return out


def detect_windows(in_dir):
    import dnd_montage as dm
    out = []
    for clip_path in find_clips(in_dir):
        try:
            dur = dm.probe_duration(clip_path)
        except Exception as ex:
            print(f"  ! skip {os.path.basename(clip_path)}: {ex}")
            continue
        for (s, e, _w, k) in dm.fight_windows(clip_path, dur):
            out.append((os.path.basename(clip_path), s, e, k))
    return out


def build_project(in_dir):
    """Clips (with suggestion windows) + a seeded cut list."""
    clips, cuts = {}, []
    labelled = windows_from_labels()
    if labelled:
        source = "labels"
        for clip, s, e, v in labelled:
            c = clips.setdefault(clip, {"clip": clip, "duration": 0.0, "windows": []})
            c["windows"].append({
                "start": round(s, 2), "end": round(e, 2),
                "is_pvp": bool(v and v.get("is_pvp")),
                "score": int(v.get("montage_score", 0)) if v else 0,
                "category": (v["categories"][0] if v and v.get("categories") else "")})
        seed = 0
        for clip, s, e, v in labelled:
            if v and v.get("is_pvp") and v.get("montage_score", 0) >= KEEP_SCORE:
                cuts.append({"id": f"seed{seed}", "clip": clip,
                             "src_in": round(v["tight_start"], 2),
                             "src_out": round(v["tight_end"], 2),
                             "category": (v["categories"][0] if v.get("categories") else ""),
                             "title": ""})
                seed += 1
    else:
        source = "detection"
        for clip, s, e, k in detect_windows(in_dir):
            c = clips.setdefault(clip, {"clip": clip, "duration": 0.0, "windows": []})
            c["windows"].append({"start": round(s, 2), "end": round(e, 2),
                                 "is_pvp": False, "score": 0,
                                 "category": "kill_win" if k else ""})
    for clip, c in clips.items():
        c["duration"] = round(_duration(os.path.join(in_dir, clip)), 2)
        c["windows"].sort(key=lambda w: w["start"])
    clip_list = sorted(clips.values(),
                       key=lambda c: recording_key(c["clip"], os.path.join(in_dir, c["clip"])))
    cuts.sort(key=lambda ct: (recording_key(ct["clip"], os.path.join(in_dir, ct["clip"])),
                              ct["src_in"]))
    for o, ct in enumerate(cuts):
        ct["order"] = o
    return {"clips_dir": in_dir, "fps": 60, "source": source,
            "clips": clip_list, "cuts": cuts}


def load_project(in_dir):
    """Fresh clips/suggestions every time; keep the user's saved cut list."""
    proj = build_project(in_dir)
    if os.path.exists(PROJECT):
        try:
            saved = json.load(open(PROJECT))
            if isinstance(saved.get("cuts"), list):
                proj["cuts"] = saved["cuts"]
        except Exception:
            pass
    return proj


def save_project(proj):
    os.makedirs(WB_DIR, exist_ok=True)
    json.dump(proj, open(PROJECT, "w"), indent=2)


def _clip_path(proj, clip):
    return os.path.join(proj["clips_dir"], clip)


# ---- media ----------------------------------------------------------------

def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300] or "ffmpeg failed")


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def remux_full(clip_path):
    """Lossless stream-copy to a browser-playable mp4 (Chrome won't play mkv)."""
    os.makedirs(FULL_DIR, exist_ok=True)
    out = os.path.join(FULL_DIR, _safe(os.path.basename(clip_path)) + ".mp4")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    try:
        _run(["ffmpeg", "-i", clip_path, "-c", "copy", "-movflags", "+faststart",
              "-y", "-v", "error", out])
    except Exception:
        _run(["ffmpeg", "-i", clip_path, "-c:v", VCODEC, "-crf", "23", "-preset",
              "veryfast", "-c:a", ACODEC, "-movflags", "+faststart", "-y", "-v", "error", out])
    return out


def waveform_data(clip_path, buckets=WAVE_BUCKETS):
    """Cached peak-envelope (0..1) of the audio, for drawing action spikes."""
    os.makedirs(WAVE_DIR, exist_ok=True)
    cache = os.path.join(WAVE_DIR, _safe(os.path.basename(clip_path)) + ".json")
    if os.path.exists(cache):
        return json.load(open(cache))
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-i", clip_path, "-ac", "1", "-ar", "8000",
                          "-f", "s16le", "-v", "quiet", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if len(a) == 0:
        peaks = [0.0] * buckets
    else:
        idx = np.linspace(0, len(a), buckets + 1).astype(int)
        peaks = [float(np.abs(a[idx[i]:idx[i + 1]]).max()) if idx[i + 1] > idx[i] else 0.0
                 for i in range(buckets)]
        m = max(peaks) or 1.0
        peaks = [round(p / m, 3) for p in peaks]
    data = {"peaks": peaks}
    json.dump(data, open(cache, "w"))
    return data


def file_url(path):
    # safe="/:" keeps the drive-letter colon literal (file:///C:/...); spaces -> %20.
    return "file:///" + quote(os.path.abspath(path).replace(os.sep, "/"), safe="/:")


def _rate(fps):
    if abs(fps - 60000 / 1001) < 0.01:
        return 60000, 1001, 1001, 60000, 60
    if abs(fps - 30000 / 1001) < 0.01:
        return 30000, 1001, 1001, 30000, 30
    if abs(fps - 24000 / 1001) < 0.01:
        return 24000, 1001, 1001, 24000, 24
    r = int(round(fps))
    return r, 1, 1, r, r


def _probe(path, entries):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", entries, "-of", "default=nw=1:nk=1", path],
                         capture_output=True, text=True).stdout.split("\n")
    return [x for x in out if x.strip()]


def _cut_meta(path):
    dur = float(_probe(path, "format=duration")[0])
    rfr = _probe(path, "stream=r_frame_rate")[0]
    num, den = (rfr.split("/") + ["1"])[:2]
    fps = float(num) / float(den or 1)
    w = int(_probe(path, "stream=width")[0]); h = int(_probe(path, "stream=height")[0])
    return dur, fps, w, h


def _tc(frame, R, offset_frames=0):
    total = frame + offset_frames
    return f"{total // (R * 3600):02d}:{total // (R * 60) % 60:02d}:{total // R % 60:02d}:{total % R:02d}"


# ---- export ---------------------------------------------------------------

def cut_exact(clip_path, src_in, src_out, out_path):
    span = max(0.05, src_out - src_in)
    _ensure_dir(out_path)
    _run(["ffmpeg", "-ss", f"{src_in:.3f}", "-i", clip_path, "-t", f"{span:.3f}",
          "-c:v", VCODEC, "-crf", CRF, "-preset", PRESET, "-c:a", ACODEC,
          "-movflags", "+faststart", "-y", "-v", "error", out_path])


def _normalize(src, out_path):
    _run(["ffmpeg", "-i", src, "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
          "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=60,setsar=1",
          "-c:v", VCODEC, "-crf", CRF, "-preset", PRESET, "-r", "60",
          "-c:a", ACODEC, "-ar", "48000", "-ac", "2", "-y", "-v", "error", out_path])


def _kept(proj):
    return sorted(proj["cuts"], key=lambda c: c.get("order", 0))


def export_folder(proj, out_dir):
    cuts_dir = os.path.join(out_dir, "cuts")
    os.makedirs(cuts_dir, exist_ok=True)
    names = []
    for i, c in enumerate(_kept(proj), 1):
        cat = c.get("category") or "clip"
        label = c.get("title") or os.path.splitext(c["clip"])[0]
        name = f"{i:02d}_{cat}_{_safe(label, 40)}.mp4"
        cut_exact(_clip_path(proj, c["clip"]), c["src_in"], c["src_out"],
                  os.path.join(cuts_dir, name))
        names.append(name)
    return names


def build_montage(proj, out_dir, intro=None, outro=None, music=None, music_vol=0.6):
    kept = _kept(proj)
    if not kept:
        raise RuntimeError("no cuts to build a montage from")
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="wb_montage_")
    try:
        segs = []
        if intro:
            n = os.path.join(tmp, "00_intro.mp4"); _normalize(intro, n); segs.append(n)
        for i, c in enumerate(kept, 1):
            seg = os.path.join(tmp, f"{i:02d}.mp4")
            cut_exact(_clip_path(proj, c["clip"]), c["src_in"], c["src_out"], seg)
            segs.append(seg)
        if outro:
            n = os.path.join(tmp, "99_outro.mp4"); _normalize(outro, n); segs.append(n)
        listfile = os.path.join(tmp, "concat.txt")
        with open(listfile, "w") as fh:
            for s in segs:
                fh.write(f"file '{os.path.abspath(s)}'\n")
        joined = os.path.join(tmp, "joined.mp4")
        _run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", listfile,
              "-c", "copy", "-movflags", "+faststart", "-y", "-v", "error", joined])
        out = os.path.join(out_dir, "montage.mp4")
        if music:
            _run(["ffmpeg", "-i", joined, "-stream_loop", "-1", "-i", music,
                  "-filter_complex",
                  f"[1:a]volume={music_vol}[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=0[a]",
                  "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", ACODEC,
                  "-shortest", "-movflags", "+faststart", "-y", "-v", "error", out])
        else:
            os.replace(joined, out)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _xml(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def write_fcpxml(cuts, out_path, title="dnd montage"):
    _d0, fps0, w, h = _cut_meta(cuts[0][0])
    rn, rd, fdn, fdd, _R = _rate(fps0)

    def frames(t):
        return int(round(t * rn / rd))

    def val(fr):
        return f"{fr * fdn}/{fdd}s"

    assets, clips, cum = [], [], 0
    for i, (path, name) in enumerate(cuts, 1):
        dur, _f, _w, _h = _cut_meta(path)
        fr = frames(dur); aid = f"a{i}"
        assets.append(
            f'    <asset id="{aid}" name="{_xml(name)}" uid="{aid}" start="0s" '
            f'duration="{val(fr)}" hasVideo="1" hasAudio="1" format="r1" '
            f'videoSources="1" audioSources="1" audioChannels="2" audioRate="48000">\n'
            f'      <media-rep kind="original-media" src="{_xml(file_url(path))}"/>\n'
            f'    </asset>')
        clips.append(
            f'            <asset-clip ref="{aid}" name="{_xml(name)}" offset="{val(cum)}" '
            f'start="0s" duration="{val(fr)}" tcFormat="NDF" audioRole="dialogue"/>')
        cum += fr
    fname = f"FFVideoFormat{w}x{h}p{int(round(fps0))}"
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n'
        '<fcpxml version="1.9">\n  <resources>\n'
        f'    <format id="r1" name="{fname}" frameDuration="{fdn}/{fdd}s" '
        f'width="{w}" height="{h}" colorSpace="1-1-1 (Rec. 709)"/>\n'
        + "\n".join(assets) + "\n  </resources>\n  <library>\n"
        f'    <event name="{_xml(title)}">\n      <project name="{_xml(title)}">\n'
        f'        <sequence format="r1" duration="{val(cum)}" tcStart="0s" '
        f'tcFormat="NDF" audioLayout="stereo" audioRate="48k">\n          <spine>\n'
        + "\n".join(clips) + "\n          </spine>\n        </sequence>\n"
        "      </project>\n    </event>\n  </library>\n</fcpxml>\n")
    open(out_path, "w", encoding="utf-8").write(xml)
    return out_path


def write_edl(cuts, out_path, title="DND_MONTAGE"):
    _d0, fps0, _w, _h = _cut_meta(cuts[0][0])
    rn, rd, _fdn, _fdd, R = _rate(fps0)
    lines = [f"TITLE: {title[:70]}", "FCM: NON-DROP FRAME"]
    reels, cum = {}, 0
    for i, (path, name) in enumerate(cuts, 1):
        dur, _f, _w, _h = _cut_meta(path)
        fr = max(1, int(round(dur * rn / rd)))
        base = os.path.basename(path)
        reel = "".join(c for c in os.path.splitext(base)[0].upper() if c.isalnum())[:8] or "CLIP"
        while reel in reels:
            reel = reel[:7] + str(len(reels) % 10)
        reels[reel] = base
        lines.append(f"{i:03d}  {reel:<8}  V     C        "
                     f"{_tc(0, R)} {_tc(fr, R)} {_tc(cum, R, 3600 * R)} {_tc(cum + fr, R, 3600 * R)}")
        lines.append(f"* FROM CLIP NAME: {base}")
        cum += fr
    open(out_path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return out_path


def export_timeline(proj, out_dir):
    names = export_folder(proj, out_dir)
    cuts = [(os.path.join(out_dir, "cuts", n), os.path.splitext(n)[0]) for n in names]
    if not cuts:
        raise RuntimeError("no cuts to export")
    fx = write_fcpxml(cuts, os.path.join(out_dir, "montage.fcpxml"))
    ed = write_edl(cuts, os.path.join(out_dir, "montage.edl"))
    return fx, ed, len(cuts)


# ---- web UI ---------------------------------------------------------------

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Clip workbench</title>
<style>
 body{font:13px system-ui,sans-serif;margin:0;background:#14161a;color:#e6e6e6}
 header{position:sticky;top:0;background:#1d2026;padding:8px 14px;border-bottom:1px solid #2b2f37;
   display:flex;gap:10px;align-items:center;flex-wrap:wrap;z-index:9}
 header b{font-size:15px}
 .clipbar{display:flex;gap:6px;overflow-x:auto;padding:8px 14px;background:#171a1f;border-bottom:1px solid #2b2f37}
 .clipbar button{white-space:nowrap;font-size:12px}
 .clipbar button.cur{background:#2563eb;border-color:#2563eb}
 .clipbar .badge{background:#2e7d32;border-radius:8px;padding:0 5px;margin-left:5px;font-size:11px}
 .main{display:flex;gap:14px;padding:14px;align-items:flex-start}
 .left{flex:1;min-width:0}.right{width:330px;flex-shrink:0}
 video{width:100%;border-radius:6px;background:#000;max-height:50vh}
 #wave{width:100%;height:140px;background:#0e1014;border-radius:6px;margin-top:6px;cursor:crosshair;display:block}
 .ctl{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px}
 button{cursor:pointer;border:1px solid #3a3f49;background:#262b33;color:#e6e6e6;border-radius:6px;padding:5px 9px}
 button:hover{border-color:#5a6270}
 .add{background:#2563eb;border-color:#2563eb;font-weight:600}
 .setin,.setout{background:#1f3a5f;border-color:#27507f}
 input,select{background:#11141a;color:#e6e6e6;border:1px solid #3a3f49;border-radius:5px;padding:5px}
 input.t{width:78px}
 .cut{background:#1b1e24;border:1px solid #2b2f37;border-radius:8px;padding:8px;margin:7px 0}
 .cut.drag{opacity:.4}.cut.over{border-color:#2563eb}
 .cut .hd{display:flex;gap:6px;align-items:center}.cut .grip{cursor:grab;color:#6b7280}
 .cut .nm{flex:1;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
 .cut .row2{display:flex;gap:6px;align-items:center;margin-top:5px}
 .cut .title{flex:1;min-width:60px}
 .exp{background:#7c3aed;border-color:#7c3aed;font-weight:600}
 .status{margin-left:6px;color:#9aa;font-size:12px}
 h3{margin:0 0 6px;font-size:13px;color:#9aa}
 .legend{font-size:11px;color:#8aa}
</style></head><body>
<header><b>Clip workbench</b>
 <span class="legend">drag on the waveform to mark a region → Add cut · loud spikes = action</span>
 <span style="flex:1"></span>
 <button class="exp" id="save">Save</button>
 <button class="exp" id="expFolder">Cuts → folder</button>
 <button class="exp" id="expTimeline">DaVinci timeline</button>
 <button class="exp" id="expMontage">Build montage</button>
 <span class="status" id="status"></span>
</header>
<div class="clipbar" id="clipbar"></div>
<div class="main">
 <div class="left">
   <video id="player" controls preload="auto"></video>
   <canvas id="wave"></canvas>
   <div class="ctl">
     <button class="setin">Set IN @ playhead</button>
     <button class="setout">Set OUT @ playhead</button>
     in <input class="t" id="inv"> out <input class="t" id="outv">
     <select id="newcat"><option value="">category…</option></select>
     <button class="add" id="addcut">+ Add cut</button>
     <span class="legend" id="seldur"></span>
   </div>
 </div>
 <div class="right">
   <h3>Montage — <span id="ncuts">0</span> cuts (drag to reorder)</h3>
   <div id="cuts"></div>
 </div>
</div>
<script>
const CATS=__CATS__;
let CLIPS=[], cur=null, wave=null, cuts=[], sel=null, dragging=false, downX=0, dragIdx=null;
const player=document.getElementById('player'), cv=document.getElementById('wave'), g=cv.getContext('2d');
const statusEl=document.getElementById('status'), enc=encodeURIComponent;
function fmt(t){t=Math.max(0,t||0);const m=Math.floor(t/60);return m+':'+(t%60).toFixed(1).padStart(4,'0');}
document.getElementById('newcat').innerHTML='<option value="">category…</option>'+CATS.map(c=>`<option>${c}</option>`).join('');

async function init(){
  CLIPS=await (await fetch('/clips')).json();
  cuts=await (await fetch('/cuts')).json();
  cuts.sort((a,b)=>(a.order||0)-(b.order||0));
  renderClipbar(); renderCuts();
  if(CLIPS.length) loadClip(CLIPS[0].clip);
  requestAnimationFrame(loop);
}
function renderClipbar(){
  const bar=document.getElementById('clipbar'); bar.innerHTML='';
  CLIPS.forEach(c=>{
    const n=cuts.filter(x=>x.clip===c.clip).length;
    const b=document.createElement('button');
    b.className=(c.clip===cur?'cur':'');
    b.innerHTML=c.clip+(n?`<span class="badge">${n}</span>`:'');
    b.onclick=()=>loadClip(c.clip); bar.appendChild(b);
  });
}
async function loadClip(clip){
  cur=clip; renderClipbar();
  player.src='/full?clip='+enc(clip);
  wave=await (await fetch('/wave?clip='+enc(clip))).json();
  sel=null; document.getElementById('seldur').textContent=''; drawWave();
}
function T2X(t){return wave?t/wave.duration*cv.width:0;}
function X2T(x){return wave?x/cv.width*wave.duration:0;}
function drawWave(){
  if(!wave) return;
  const W=cv.clientWidth, H=140; cv.width=W; cv.height=H;
  g.clearRect(0,0,W,H);
  // suggestion windows
  (wave.windows||[]).forEach(w=>{const x0=T2X(w.start),x1=T2X(w.end);
    g.fillStyle=w.is_pvp?'rgba(110,192,112,.13)':'rgba(201,162,39,.10)';
    g.fillRect(x0,0,x1-x0,H);
    g.fillStyle=w.is_pvp?'rgba(110,192,112,.6)':'rgba(201,162,39,.5)';
    g.fillRect(x0,0,Math.max(1,x1-x0),2);});
  // peaks
  const p=wave.peaks, n=p.length; g.fillStyle='#3b6fb0';
  for(let i=0;i<n;i++){const x=i/n*W, h=p[i]*(H-6); g.fillRect(x,H-h,Math.max(1,W/n),h);}
  // existing cuts for this clip
  cuts.filter(c=>c.clip===cur).forEach(c=>{const x0=T2X(c.src_in),x1=T2X(c.src_out);
    g.fillStyle='rgba(37,99,235,.28)'; g.fillRect(x0,0,x1-x0,H);
    g.strokeStyle='#2563eb'; g.strokeRect(x0+.5,.5,x1-x0-1,H-1);});
  // selection
  if(sel){const x0=T2X(Math.min(sel.a,sel.b)),x1=T2X(Math.max(sel.a,sel.b));
    g.fillStyle='rgba(255,255,255,.18)'; g.fillRect(x0,0,x1-x0,H);}
  // playhead
  const px=T2X(player.currentTime); g.strokeStyle='#e23b3b'; g.lineWidth=1.5;
  g.beginPath(); g.moveTo(px,0); g.lineTo(px,H); g.stroke();
}
function loop(){ if(wave && !player.paused) drawWave(); requestAnimationFrame(loop); }
player.addEventListener('seeked',drawWave); player.addEventListener('timeupdate',drawWave);
cv.addEventListener('mousedown',e=>{const r=cv.getBoundingClientRect();downX=e.clientX-r.left;dragging=true;sel={a:X2T(downX),b:X2T(downX)};});
cv.addEventListener('mousemove',e=>{if(!dragging)return;const r=cv.getBoundingClientRect();sel.b=X2T(e.clientX-r.left);
  document.getElementById('seldur').textContent='selection '+fmt(Math.abs(sel.b-sel.a));drawWave();});
window.addEventListener('mouseup',e=>{if(!dragging)return;dragging=false;const r=cv.getBoundingClientRect();const upX=e.clientX-r.left;
  if(Math.abs(upX-downX)<4){player.currentTime=X2T(upX);sel=null;document.getElementById('seldur').textContent='';}
  else{const a=Math.min(sel.a,sel.b),b=Math.max(sel.a,sel.b);sel={a,b};document.getElementById('inv').value=a.toFixed(2);document.getElementById('outv').value=b.toFixed(2);}
  drawWave();});
document.querySelector('.setin').onclick=()=>{document.getElementById('inv').value=player.currentTime.toFixed(2);};
document.querySelector('.setout').onclick=()=>{document.getElementById('outv').value=player.currentTime.toFixed(2);};
document.getElementById('addcut').onclick=()=>{
  let a=parseFloat(document.getElementById('inv').value), b=parseFloat(document.getElementById('outv').value);
  if(sel){a=Math.min(sel.a,sel.b);b=Math.max(sel.a,sel.b);}
  if(isNaN(a)||isNaN(b)||b-a<0.1){statusEl.textContent='set a valid in/out (or drag a region)';return;}
  cuts.push({id:'c'+Date.now()+Math.floor(Math.random()*1000),clip:cur,src_in:+a.toFixed(2),src_out:+b.toFixed(2),
    category:document.getElementById('newcat').value,title:''});
  sel=null;document.getElementById('inv').value='';document.getElementById('outv').value='';document.getElementById('seldur').textContent='';
  renderCuts();renderClipbar();drawWave();save();
};
function renderCuts(){
  cuts.forEach((c,i)=>c.order=i);
  const box=document.getElementById('cuts'); box.innerHTML='';
  document.getElementById('ncuts').textContent=cuts.length;
  cuts.forEach((c,idx)=>{
    const d=document.createElement('div'); d.className='cut'; d.draggable=true;
    d.innerHTML=`<div class="hd"><span class="grip">⠿</span><span class="nm">${idx+1}. ${c.clip}</span>
       <button class="jump">▶</button><button class="del">✕</button></div>
       <div class="row2">[${c.src_in.toFixed(1)}–${c.src_out.toFixed(1)}] ${fmt(c.src_out-c.src_in)}
       <select class="cat"><option value="">—</option>${CATS.map(x=>`<option ${c.category===x?'selected':''}>${x}</option>`).join('')}</select>
       <input class="title" placeholder="title" value="${(c.title||'').replace(/"/g,'&quot;')}"></div>`;
    const q=s=>d.querySelector(s);
    q('.jump').onclick=async()=>{if(cur!==c.clip)await loadClip(c.clip);player.currentTime=c.src_in;};
    q('.del').onclick=()=>{cuts.splice(idx,1);renderCuts();renderClipbar();drawWave();save();};
    q('.cat').onchange=e=>{c.category=e.target.value;save();};
    q('.title').onchange=e=>{c.title=e.target.value;save();};
    d.ondragstart=()=>{dragIdx=idx;d.classList.add('drag');};
    d.ondragend=()=>{d.classList.remove('drag');document.querySelectorAll('.cut').forEach(x=>x.classList.remove('over'));};
    d.ondragover=e=>{e.preventDefault();d.classList.add('over');};
    d.ondragleave=()=>d.classList.remove('over');
    d.ondrop=e=>{e.preventDefault();if(dragIdx===null||dragIdx===idx)return;
      const [m]=cuts.splice(dragIdx,1);cuts.splice(idx,0,m);dragIdx=null;renderCuts();drawWave();save();};
    box.appendChild(d);
  });
}
async function save(){cuts.forEach((c,i)=>c.order=i);
  await fetch('/cuts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cuts)});}
async function exp(path,confirmMsg){
  if(confirmMsg && !confirm(confirmMsg))return;
  cuts.forEach((c,i)=>c.order=i); statusEl.textContent='working… (encoding)';
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cuts)});
  const j=await r.json(); statusEl.textContent=j.msg||(j.ok?'done':'error');
}
document.getElementById('save').onclick=async()=>{await save();statusEl.textContent='saved';};
document.getElementById('expFolder').onclick=()=>exp('/export_folder');
document.getElementById('expTimeline').onclick=()=>exp('/export_timeline');
document.getElementById('expMontage').onclick=()=>exp('/export_montage','Build montage from these cuts? (re-encodes, can take a few min)');
window.addEventListener('resize',drawWave);
init();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    proj = None
    out_dir = "montage"
    intro = outro = music = None

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_range(self, path, ctype="video/mp4"):
        size = os.path.getsize(path)
        rng = self.headers.get("Range", "")
        start, end = 0, size - 1
        partial = rng.startswith("bytes=")
        if partial:
            s, _, e = rng[6:].partition("-")
            start = int(s) if s else 0
            end = int(e) if e else size - 1
            end = min(end, size - 1)
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(262144, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _clip_arg(self):
        clip = parse_qs(urlparse(self.path).query).get("clip", [""])[0]
        if clip not in {c["clip"] for c in self.proj["clips"]}:
            return None
        return clip

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE.replace("__CATS__", json.dumps(CATEGORIES)),
                              "text/html; charset=utf-8")
        if u.path == "/clips":
            return self._send(200, [{"clip": c["clip"], "duration": c["duration"],
                                     "n_windows": len(c["windows"])} for c in self.proj["clips"]])
        if u.path == "/cuts":
            return self._send(200, self.proj["cuts"])
        if u.path == "/wave":
            clip = self._clip_arg()
            if not clip:
                return self._send(404, {"error": "unknown clip"})
            c = next(x for x in self.proj["clips"] if x["clip"] == clip)
            wf = waveform_data(_clip_path(self.proj, clip))
            return self._send(200, {"duration": c["duration"], "peaks": wf["peaks"],
                                    "windows": c["windows"]})
        if u.path == "/full":
            clip = self._clip_arg()
            if not clip:
                return self._send(404, {"error": "unknown clip"})
            return self._serve_range(remux_full(_clip_path(self.proj, clip)))
        self._send(404, {"error": "not found"})

    def _set_cuts(self):
        n = int(self.headers.get("Content-Length", 0))
        items = json.loads(self.rfile.read(n).decode()) if n else []
        clean = []
        for i, c in enumerate(items):
            cat = c.get("category", "")
            clean.append({"id": c.get("id", f"c{i}"), "clip": c["clip"],
                          "src_in": round(float(c["src_in"]), 3),
                          "src_out": round(float(c["src_out"]), 3),
                          "category": cat if cat in CATEGORIES else "",
                          "title": (c.get("title") or "").strip(), "order": i})
        self.proj["cuts"] = clean
        save_project(self.proj)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            self._set_cuts()
            if path == "/cuts":
                return self._send(200, {"ok": True, "msg": "saved"})
            if path == "/export_folder":
                names = export_folder(self.proj, self.out_dir)
                return self._send(200, {"ok": True,
                    "msg": f"exported {len(names)} cut(s) to {self.out_dir}/cuts/"})
            if path == "/export_timeline":
                fx, _ed, n = export_timeline(self.proj, self.out_dir)
                return self._send(200, {"ok": True,
                    "msg": f"wrote {os.path.basename(fx)} (+ .edl), {n} clips — import the .fcpxml in Resolve"})
            if path == "/export_montage":
                out = build_montage(self.proj, self.out_dir, self.intro, self.outro, self.music)
                return self._send(200, {"ok": True, "msg": f"montage built: {out}"})
            self._send(404, {"ok": False, "msg": "unknown action"})
        except Exception as ex:
            self._send(200, {"ok": False, "msg": f"error: {ex}"})


def main():
    ap = argparse.ArgumentParser(description="Fast clip workbench (full clips + waveform + cuts)")
    ap.add_argument("--in", dest="in_dir", default="clips", help="folder with the SOURCE clips")
    ap.add_argument("--out", default="montage", help="output folder for exports")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--intro", default=None)
    ap.add_argument("--outro", default=None)
    ap.add_argument("--music", default=None)
    ap.add_argument("--rebuild", action="store_true", help="rebuild clips/suggestions (keeps saved cuts)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    proj = load_project(args.in_dir)
    if not proj["clips"]:
        sys.exit("No clips with detected windows. Run detection/labeling first, or check --in.")
    save_project(proj)
    print(f"Project: {len(proj['clips'])} clip(s), {len(proj['cuts'])} seeded cut(s) "
          f"(from {proj.get('source')}). Full clips + waveforms load on demand.")

    Handler.proj = proj
    Handler.out_dir = args.out
    Handler.intro, Handler.outro, Handler.music = args.intro, args.outro, args.music
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\nWorkbench: {url}   (exports -> {args.out}/)")
    print("Pick a clip, watch the waveform for spikes, drag a region, Add cut, reorder, export. Ctrl-C to stop.")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        save_project(proj)
        print("\nStopped. Project saved to", PROJECT)


if __name__ == "__main__":
    main()
