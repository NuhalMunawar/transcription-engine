"""
Web-based download progress viewer for nyrahealth/CrisperWhisper.
Opens a browser page showing a live progress bar + ETA.
Run: .venv\Scripts\python.exe download_progress.py
"""

import os, sys, threading, time, json, webbrowser
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from huggingface_hub import HfApi, snapshot_download

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_ID   = "nyrahealth/CrisperWhisper"
HF_TOKEN   = os.getenv("HF_TOKEN")
CACHE_ROOT = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
BLOBS_DIR  = CACHE_ROOT / "models--nyrahealth--CrisperWhisper" / "blobs"
PORT       = 7861

# ── State shared between download thread and SSE ───────────────────────────
state = {
    "total_bytes":    0,
    "done":           False,
    "error":          None,
    "start_time":     None,
}

# ── Get total repo size upfront ────────────────────────────────────────────
def _get_total_size() -> int:
    try:
        api   = HfApi()
        files = list(api.list_repo_tree(MODEL_ID, token=HF_TOKEN, recursive=True))
        return sum(getattr(f, "size", 0) or 0 for f in files)
    except Exception:
        return 3_220_000_000   # fallback ~3.22 GB

# ── Background download thread ─────────────────────────────────────────────
def _download():
    state["start_time"] = time.time()
    try:
        snapshot_download(
            MODEL_ID,
            token=HF_TOKEN,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
        )
        state["done"] = True
    except Exception as e:
        state["error"] = str(e)

def _get_downloaded_bytes() -> int:
    if not BLOBS_DIR.exists():
        return 0
    return sum(f.stat().st_size for f in BLOBS_DIR.iterdir() if f.is_file())

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Download — CrisperWhisper</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f1117;
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
  }
  .card {
    background: #1a1d2e;
    border: 1px solid #2d3148;
    border-radius: 16px;
    padding: 40px 48px;
    width: 580px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
  }
  .logo { font-size: 32px; margin-bottom: 8px; }
  h1 { font-size: 20px; font-weight: 700; color: #f0f4ff; margin-bottom: 4px; }
  .sub { font-size: 13px; color: #6b7280; margin-bottom: 32px; }

  .bar-wrap {
    background: #262a3f;
    border-radius: 99px;
    height: 14px;
    overflow: hidden;
    margin-bottom: 16px;
  }
  .bar {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa);
    transition: width 0.6s ease;
    width: 0%;
    position: relative;
  }
  .bar.pulse::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%);
    animation: shimmer 1.4s infinite;
  }
  @keyframes shimmer {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }

  .stats {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #9ca3af;
    margin-bottom: 28px;
  }
  .stats span strong { color: #e2e8f0; }

  .files-label { font-size: 12px; color: #6b7280; margin-bottom: 8px; text-transform: uppercase; letter-spacing: .05em; }
  .log {
    background: #0f1117;
    border: 1px solid #1f2333;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 12px;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    color: #7c83a0;
    height: 110px;
    overflow-y: auto;
    line-height: 1.7;
  }
  .log .entry { color: #a5b4fc; }

  .done-box {
    display: none;
    text-align: center;
    padding: 24px;
    background: #0f2918;
    border: 1px solid #16a34a44;
    border-radius: 12px;
    margin-top: 24px;
  }
  .done-box.show { display: block; }
  .done-box .checkmark { font-size: 40px; margin-bottom: 8px; }
  .done-box p { color: #4ade80; font-size: 15px; font-weight: 600; }
  .done-box small { color: #6b7280; font-size: 12px; }

  .err-box {
    display: none;
    padding: 16px;
    background: #2d0f0f;
    border: 1px solid #991b1b44;
    border-radius: 8px;
    color: #f87171;
    font-size: 13px;
    margin-top: 16px;
  }
  .err-box.show { display: block; }
</style>
</head>
<body>
<div class="card">
  <div class="logo">&#x1F9E0;</div>
  <h1>Downloading CrisperWhisper</h1>
  <p class="sub">nyrahealth/CrisperWhisper &nbsp;&bull;&nbsp; ~3.2 GB &nbsp;&bull;&nbsp; Saved permanently to HF cache</p>

  <div class="bar-wrap"><div class="bar pulse" id="bar"></div></div>

  <div class="stats">
    <span><strong id="pct">0%</strong> complete</span>
    <span><strong id="dl">0 MB</strong> / <strong id="tot">–</strong></span>
    <span>Speed: <strong id="spd">–</strong></span>
    <span>ETA: <strong id="eta">–</strong></span>
  </div>

  <div class="files-label">Live log</div>
  <div class="log" id="log"><span style="color:#4b5563">Waiting for download thread...</span></div>

  <div class="done-box" id="donebox">
    <div class="checkmark">&#x2705;</div>
    <p>Download complete!</p>
    <small>Model cached — you can close this window and run the pipeline.</small>
  </div>
  <div class="err-box" id="errbox"></div>
</div>

<script>
const bar    = document.getElementById('bar');
const pctEl  = document.getElementById('pct');
const dlEl   = document.getElementById('dl');
const totEl  = document.getElementById('tot');
const spdEl  = document.getElementById('spd');
const etaEl  = document.getElementById('eta');
const logEl  = document.getElementById('log');
const doneEl = document.getElementById('donebox');
const errEl  = document.getElementById('errbox');

function fmt(bytes) {
  if (bytes < 1024**2)        return (bytes/1024).toFixed(1)+' KB';
  if (bytes < 1024**3)        return (bytes/1024**2).toFixed(1)+' MB';
  return (bytes/1024**3).toFixed(2)+' GB';
}
function fmtEta(secs) {
  if (!isFinite(secs) || secs <= 0) return '–';
  const m = Math.floor(secs/60), s = Math.floor(secs%60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}
function addLog(msg) {
  const d = document.createElement('div');
  d.className = 'entry';
  d.textContent = msg;
  logEl.appendChild(d);
  logEl.scrollTop = logEl.scrollHeight;
}

const es = new EventSource('/progress');
es.onmessage = function(e) {
  const d = JSON.parse(e.data);
  if (d.error) {
    errEl.textContent = 'Error: ' + d.error;
    errEl.classList.add('show');
    es.close(); return;
  }
  const pct = d.total > 0 ? Math.min(100, (d.downloaded / d.total) * 100) : 0;
  bar.style.width = pct.toFixed(1) + '%';
  pctEl.textContent = pct.toFixed(1) + '%';
  dlEl.textContent  = fmt(d.downloaded);
  totEl.textContent = d.total > 0 ? fmt(d.total) : '–';
  spdEl.textContent = d.speed > 0 ? fmt(d.speed)+'/s' : '–';
  etaEl.textContent = fmtEta(d.eta);
  if (d.log) addLog(d.log);
  if (d.done) {
    bar.style.width = '100%';
    bar.classList.remove('pulse');
    pctEl.textContent = '100%';
    etaEl.textContent = 'Done';
    doneEl.classList.add('show');
    es.close();
  }
};
es.onerror = function() {
  addLog('[connection lost — page will reconnect]');
};
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.get("/progress")
def progress():
    def _stream():
        prev_bytes  = _get_downloaded_bytes()
        prev_time   = time.time()
        total       = state["total_bytes"]
        log_counter = 0

        while True:
            time.sleep(1)
            now   = time.time()
            dl    = _get_downloaded_bytes()
            delta = now - prev_time
            speed = max(0, (dl - prev_bytes) / delta) if delta > 0 else 0
            prev_bytes = dl
            prev_time  = now
            remaining  = total - dl
            eta        = (remaining / speed) if speed > 0 else float("inf")

            pct = (dl / total * 100) if total > 0 else 0
            log_counter += 1
            log_msg = None
            if log_counter % 5 == 0:
                log_msg = f"{pct:.1f}% — {dl/1024**3:.2f} GB / {total/1024**3:.2f} GB  ({speed/1024**2:.1f} MB/s)"

            payload = {
                "downloaded": dl,
                "total":      total,
                "speed":      speed,
                "eta":        eta if eta != float("inf") else -1,
                "done":       state["done"],
                "error":      state["error"],
                "log":        log_msg,
            }
            yield f"data: {json.dumps(payload)}\n\n"

            if state["done"] or state["error"]:
                break

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching repo file sizes...")
    total = _get_total_size()
    state["total_bytes"] = total
    print(f"Total to download: {total/1024**3:.2f} GB")

    print("Starting download thread...")
    t = threading.Thread(target=_download, daemon=True)
    t.start()

    print(f"\nOpening browser at http://localhost:{PORT}")
    print("Keep this terminal open until the download finishes.\n")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
