#!/usr/bin/env python3
"""
review.py — local human-labeling UI for candidate windows (free ground truth).

The paid Opus seed is the design's ground-truth source, but it costs money and the
local 7B judge can't be trusted blind (it misses PvP). This tool lets you label the
candidate windows yourself — the highest-trust source in labels.py — for free.

It reads the windows already in .labels/ (written by `seed.py --judge local`), cuts
each one to a short MP4 *with audio* (your voice callouts are a strong PvP cue),
serves a labeling page, and writes your verdicts back as source="human" — which
overrides the local model's guess for that window. These human labels are what
calibrate.py compares the local teacher against, and what the student trains on.

Stdlib only (no Flask): a threaded http.server + a generated page.

Usage (clips must be reachable to cut the previews):
    python review.py --in "C:\\Users\\Frank\\Videos"
    # then open http://localhost:8000  (it tries to open your browser)
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import labels as label_store
from judge import CATEGORIES, Verdict

REVIEW_DIR = ".review"          # cached window preview MP4s
PREVIEW_WIDTH = 854             # 480p-ish; small + fast to encode, enough to judge


def _wid(clip, start, end):
    """Stable id for a window, safe as a filename."""
    base = f"{clip}__{start:.1f}-{end:.1f}"
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in base)


def load_windows(in_dir):
    """Every window in .labels/, with its local suggestion + any existing human
    label, and the resolved source-clip path (for cutting the preview)."""
    out = []
    for p in sorted(glob.glob(os.path.join(label_store.LABELS_DIR, "*.json"))):
        data = json.load(open(p))
        clip = data["clip"]
        clip_path = os.path.join(in_dir, clip)
        # collapse records for the same window: prefer human, fall back to local/etc.
        by_window = {}
        for r in data["records"]:
            key = (round(r["window"][0], 1), round(r["window"][1], 1))
            cur = by_window.get(key)
            if cur is None or r["source"] == "human":
                by_window[key] = r
        for (s, e), r in by_window.items():
            start, end = r["window"]
            v = r["verdict"]
            human = v if r["source"] == "human" else None
            suggestion = None if r["source"] == "human" else v
            out.append({
                "id": _wid(clip, start, end),
                "clip": clip, "clip_path": clip_path,
                "start": start, "end": end, "dur": round(end - start, 1),
                "frames_used": r.get("frames_used", 9),
                "suggestion": suggestion, "human": human,
                "exists": os.path.exists(clip_path),
            })
    out.sort(key=lambda w: (w["clip"], w["start"]))
    return out


def cut_preview(clip_path, start, end, out_path, width=PREVIEW_WIDTH):
    """Encode [start, end] to a small, seekable MP4 with audio. Skips if cached."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True
    span = max(0.1, end - start)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-ss", f"{start:.3f}", "-i", clip_path, "-t", f"{span:.3f}",
         "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "28", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "96k",
         "-y", "-v", "error", out_path],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ! ffmpeg failed for {os.path.basename(clip_path)} "
              f"[{start:.1f}-{end:.1f}]: {r.stderr.strip()[:200]}")
        return False
    return True


# ---- page -----------------------------------------------------------------

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>DnD window review</title>
<style>
  body{font:14px system-ui,sans-serif;margin:0;background:#14161a;color:#e6e6e6}
  header{position:sticky;top:0;background:#1d2026;padding:12px 20px;border-bottom:1px solid #2b2f37;
         display:flex;gap:18px;align-items:center;z-index:5}
  header b{font-size:16px} .prog{color:#8ad}
  .wrap{max-width:900px;margin:0 auto;padding:18px}
  .card{background:#1b1e24;border:1px solid #2b2f37;border-radius:10px;padding:14px;margin:14px 0}
  .card.done{border-color:#2e7d32}
  .card h3{margin:0 0 4px;font-size:15px} .meta{color:#9aa;font-size:12px;margin-bottom:8px}
  video{width:100%;border-radius:6px;background:#000;max-height:420px}
  .sugg{font-size:12px;color:#c9a227;margin:8px 0}
  .row{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:8px 0}
  .row label{color:#9aa;margin-right:6px}
  button{cursor:pointer;border:1px solid #3a3f49;background:#262b33;color:#e6e6e6;border-radius:6px;padding:6px 10px}
  button.sel{outline:2px solid #8ad} .pvp.yes.sel{background:#2e7d32;border-color:#2e7d32}
  .accept{background:#2e7d32;border-color:#2e7d32}
  .pvp.no.sel{background:#b23b3b;border-color:#b23b3b}
  .score button{min-width:30px;padding:5px 0;width:30px;text-align:center}
  .cats button.sel{background:#3949ab;border-color:#3949ab}
  input[type=number]{width:64px;background:#11141a;color:#e6e6e6;border:1px solid #3a3f49;border-radius:5px;padding:4px}
  input[type=text]{flex:1;min-width:200px;background:#11141a;color:#e6e6e6;border:1px solid #3a3f49;border-radius:5px;padding:6px}
  .save{background:#2563eb;border-color:#2563eb;font-weight:600;padding:8px 16px}
  .status{color:#6cc070;font-size:12px;margin-left:10px}
</style></head><body>
<header><b>DnD window review</b>
  <span class="prog"><span id="done">0</span> / <span id="total">0</span> labeled</span>
  <span style="color:#8aa;font-size:12px">PvP = an enemy <i>player</i> is involved. Score 0-10 keep-worthiness. Yellow = local model's guess.</span>
</header>
<div class="wrap" id="wrap"></div>
<script>
const WINDOWS = __WINDOWS__;
const CATS = __CATS__;
let doneCount = 0;
const wrap = document.getElementById('wrap');
document.getElementById('total').textContent = WINDOWS.length;

function badge(s){ if(!s) return ''; return `local guess: pvp=${s.is_pvp} score=${s.montage_score}`+
  (s.categories&&s.categories.length?` [${s.categories.join(', ')}]`:'')+(s.reason?` — ${s.reason}`:''); }

WINDOWS.forEach(w=>{
  const init = w.human || w.suggestion || {is_pvp:null,montage_score:null,categories:[],
     tight_start:w.start,tight_end:w.end,reason:''};
  const st = {is_pvp: w.human? w.human.is_pvp : null,
              score: w.human? w.human.montage_score : null,
              cats: new Set(w.human? (w.human.categories||[]) : []),
              ts: (init.tight_start!=null?init.tight_start:w.start),
              te: (init.tight_end!=null?init.tight_end:w.end)};
  const card = document.createElement('div'); card.className='card'+(w.human?' done':'');
  if(w.human) doneCount++;
  card.innerHTML = `
    <h3>${w.clip}</h3>
    <div class="meta">[${w.start}s – ${w.end}s] · ${w.dur}s${w.exists?'':' · <span style="color:#e66">clip not found in --in</span>'}</div>
    ${w.exists?`<video controls preload="metadata" src="/video?id=${encodeURIComponent(w.id)}"></video>`:''}
    <div class="sugg">${badge(w.suggestion)}</div>
    <div class="row"><label>PvP?</label>
      <button class="pvp yes">Yes</button><button class="pvp no">No</button></div>
    <div class="row score"><label>Score</label>${[...Array(11).keys()].map(i=>`<button data-s="${i}">${i}</button>`).join('')}</div>
    <div class="row cats"><label>Categories</label>${CATS.map(c=>`<button data-c="${c}">${c}</button>`).join('')}</div>
    <div class="row"><label>Trim</label>start <input type="number" step="0.1" class="ts" value="${st.ts}">
       end <input type="number" step="0.1" class="te" value="${st.te}"></div>
    <div class="row"><input type="text" class="reason" placeholder="one-line reason (optional)" value="${(w.human&&w.human.reason)||''}"></div>
    <div class="row">${w.suggestion?'<button class="accept">✓ Accept local guess</button>':''}<button class="save">Save human label</button><span class="status"></span></div>`;
  wrap.appendChild(card);
  const q=s=>card.querySelector(s), qa=s=>[...card.querySelectorAll(s)];
  const yes=q('.pvp.yes'), no=q('.pvp.no');
  function paintPvp(){ yes.classList.toggle('sel',st.is_pvp===true); no.classList.toggle('sel',st.is_pvp===false); }
  yes.onclick=()=>{st.is_pvp=true;paintPvp()}; no.onclick=()=>{st.is_pvp=false;paintPvp()};
  qa('.score button').forEach(b=>b.onclick=()=>{st.score=+b.dataset.s;
     qa('.score button').forEach(x=>x.classList.toggle('sel',+x.dataset.s===st.score))});
  qa('.cats button').forEach(b=>b.onclick=()=>{const c=b.dataset.c;
     if(st.cats.has(c))st.cats.delete(c); else st.cats.add(c); b.classList.toggle('sel',st.cats.has(c))});
  // paint initial state
  paintPvp();
  if(st.score!=null) qa('.score button').forEach(x=>x.classList.toggle('sel',+x.dataset.s===st.score));
  qa('.cats button').forEach(b=>b.classList.toggle('sel',st.cats.has(b.dataset.c)));

  async function doSave(){
    const status=q('.status');
    if(st.is_pvp===null){status.textContent='pick PvP yes/no first';status.style.color='#e88';return;}
    if(st.score===null){status.textContent='pick a score 0-10';status.style.color='#e88';return;}
    const body={id:w.id, is_pvp:st.is_pvp, montage_score:st.score, categories:[...st.cats],
      tight_start:+q('.ts').value, tight_end:+q('.te').value, reason:q('.reason').value};
    const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    if(j.ok){ if(!card.classList.contains('done')){card.classList.add('done');doneCount++;}
      document.getElementById('done').textContent=doneCount;
      status.style.color='#6cc070'; status.textContent='saved ✓'; }
    else { status.style.color='#e88'; status.textContent='error: '+(j.error||'?'); }
  }
  q('.save').onclick=doSave;
  const acc=q('.accept');
  if(acc) acc.onclick=()=>{                       // one click: take the local guess as-is + save
    const s=w.suggestion;
    st.is_pvp=s.is_pvp; st.score=s.montage_score; st.cats=new Set(s.categories||[]);
    paintPvp();
    qa('.score button').forEach(x=>x.classList.toggle('sel',+x.dataset.s===st.score));
    qa('.cats button').forEach(b=>b.classList.toggle('sel',st.cats.has(b.dataset.c)));
    q('.reason').value=s.reason||'local correct';
    doSave();
  };
});
document.getElementById('done').textContent=doneCount;
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    index = {}          # wid -> window dict (set on the class before serving)

    def log_message(self, *a):
        pass            # quiet

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            wins = [{k: w[k] for k in ("id", "clip", "start", "end", "dur",
                                       "suggestion", "human", "exists")}
                    for w in self.index.values()]
            page = (PAGE.replace("__WINDOWS__", json.dumps(wins))
                        .replace("__CATS__", json.dumps(CATEGORIES)))
            return self._send(200, page, "text/html; charset=utf-8")
        if u.path == "/video":
            wid = parse_qs(u.query).get("id", [""])[0]
            w = self.index.get(wid)
            if not w:
                return self._send(404, {"error": "unknown id"})
            path = os.path.join(REVIEW_DIR, wid + ".mp4")
            if not os.path.exists(path):
                return self._send(404, {"error": "preview not encoded"})
            data = open(path, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "none")
            self.end_headers()
            return self.wfile.write(data)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/save":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n).decode())
            w = self.index[body["id"]]
            cats = [c for c in body.get("categories", []) if c in CATEGORIES]
            verdict = Verdict(
                is_pvp=bool(body["is_pvp"]),
                categories=cats,
                montage_score=int(body["montage_score"]),
                tight_start=float(body.get("tight_start", w["start"])),
                tight_end=float(body.get("tight_end", w["end"])),
                confidence=1.0,
                reason=(body.get("reason") or "").strip() or "human review",
            )
            label_store.save_verdict(w["clip"], (w["start"], w["end"]),
                                     w["frames_used"], verdict.to_dict(), source="human")
            w["human"] = verdict.to_dict()
            self._send(200, {"ok": True})
        except Exception as ex:
            self._send(200, {"ok": False, "error": str(ex)})


def main():
    ap = argparse.ArgumentParser(description="Human-label candidate windows (free ground truth)")
    ap.add_argument("--in", dest="in_dir", default="clips",
                    help="folder with the SOURCE clips (to cut window previews)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--width", type=int, default=PREVIEW_WIDTH)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    windows = load_windows(args.in_dir)
    if not windows:
        sys.exit("No windows in .labels/ — run `python seed.py --in <clips> --judge local` first.")

    missing = [w for w in windows if not w["exists"]]
    if missing:
        print(f"! {len(missing)} window(s) have no source clip in {args.in_dir} "
              f"(no preview; you can still label, but blind).")

    print(f"Encoding {len(windows) - len(missing)} window preview(s) to {REVIEW_DIR}/ ...")
    done = 0
    for w in windows:
        if not w["exists"]:
            continue
        if cut_preview(w["clip_path"], w["start"], w["end"],
                       os.path.join(REVIEW_DIR, w["id"] + ".mp4"), args.width):
            done += 1
            print(f"  [{done}] {w['clip']} [{w['start']:.1f}-{w['end']:.1f}]")

    Handler.index = {w["id"]: w for w in windows}
    already = sum(1 for w in windows if w["human"])
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\nReview server: {url}   ({len(windows)} windows, {already} already labeled)")
    print("Label PvP + score for each, click Save. Ctrl-C here when done.")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Human labels saved under .labels/ (source=human, top trust).")


if __name__ == "__main__":
    main()
