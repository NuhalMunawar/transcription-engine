#!/usr/bin/env python3
"""
Emotion-Aware Transcription — Web UI
=====================================
Run:  python app.py
Opens http://localhost:7860 automatically in your browser.

Drag-and-drop (or click) an audio file to start.
Live progress is streamed back step by step.
"""

import io
import json
import os
import queue
import signal
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Dict

# Suppress Windows symlinks warning from huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    import uvicorn
except ImportError:
    sys.exit(
        "Missing web dependencies. Run:\n"
        '  .venv\\Scripts\\pip install fastapi "uvicorn[standard]"'
    )

# ── Import pipeline pieces from main.py ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from main import (
    _load_audio,
    _run_pyannote_vad,
    build_segments,
    _load_asr_pipeline,
    transcribe_segment,
    formatting_pass,
    mark_unfinished,
    analyse_emotions_hume,
    render_html,
    _palette,
)

# ── Thread-local stdout router ───────────────────────────────────────────────
# Replaces sys.stdout so print() calls inside the pipeline functions are routed
# to the correct per-job SSE queue rather than the console.

_tl = threading.local()          # .write_fn = callable | None
_orig_stdout = sys.stdout


class _PipelineStdout:
    """Forwards writes to the per-thread queue writer, or original stdout."""

    def write(self, text: str) -> int:
        fn = getattr(_tl, "write_fn", None)
        if fn is not None:
            fn(text)
        else:
            _orig_stdout.write(text)
        return len(text)

    def flush(self):
        _orig_stdout.flush()

    def isatty(self):
        return False


sys.stdout = _PipelineStdout()


def _make_log_fn(q: queue.Queue, step: str):
    """Returns a write() function that buffers lines and sends them as SSE logs."""
    buf = [""]

    def fn(text: str):
        buf[0] += text
        while "\n" in buf[0]:
            line, buf[0] = buf[0].split("\n", 1)
            line = line.strip("\r").strip()
            if line:
                q.put_nowait({"type": "log", "step": step, "msg": line})

    return fn


# ── Job store ────────────────────────────────────────────────────────────────
_jobs: Dict[str, dict] = {}

# Cache the ASR pipeline (heavy, ~1.5 GB) so it only loads once per server run
_asr_pipe = None
_asr_lock = threading.Lock()


def _get_asr_pipe():
    global _asr_pipe
    with _asr_lock:
        if _asr_pipe is None:
            _asr_pipe = _load_asr_pipeline()
    return _asr_pipe


# ── Worker (runs in background thread) ───────────────────────────────────────
def _worker(job_id: str, audio_path: str, hume_key: str):
    job = _jobs[job_id]
    q: queue.Queue = job["q"]

    def send(obj: dict):
        q.put_nowait(obj)

    def begin_step(step: str, label: str):
        send({"type": "step_start", "step": step, "label": label})
        _tl.write_fn = _make_log_fn(q, step)

    def end_step(step: str):
        send({"type": "step_done", "step": step})
        _tl.write_fn = None

    try:
        # ── Load audio ────────────────────────────────────────────────────
        begin_step("load", "Loading Audio")
        audio = _load_audio(audio_path)
        end_step("load")

        # ── Voice Activity Detection ──────────────────────────────────────
        begin_step("vad", "Voice Activity Detection")
        spans = _run_pyannote_vad(audio)
        segments = build_segments(spans, audio)
        end_step("vad")

        if not segments:
            send({"type": "error", "msg": "No speech detected in the audio file."})
            return

        # ── CrisperWhisper transcription ──────────────────────────────────
        begin_step("asr", "CrisperWhisper Transcription")
        pipe = _get_asr_pipe()

        for i, seg in enumerate(segments):
            raw = transcribe_segment(pipe, seg)
            seg.raw_text = raw
            seg.formatted_text = formatting_pass(raw)
            mark_unfinished(seg)
            send({
                "type": "segment_transcribed",
                "n": i + 1,
                "total": len(segments),
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.formatted_text,
            })

        end_step("asr")

        # ── Hume AI emotion analysis ──────────────────────────────────────
        begin_step("hume", "Hume AI Emotion Analysis")
        analyse_emotions_hume(segments, audio_path, hume_key)
        end_step("hume")

        # ── Package result ────────────────────────────────────────────────
        result_segs = [
            {
                "n": i + 1,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "emotion": seg.top_emotion,
                "color": seg.emotion_color_hex,
                "text": seg.formatted_text,
                "score": seg.emotion_score,
                "unfinished": seg.unfinished,
            }
            for i, seg in enumerate(segments)
        ]
        html_out = render_html(segments, audio_path)
        job["result"] = {"segments": result_segs, "html": html_out}
        send({"type": "done", "segments": result_segs})

    except Exception as exc:
        import traceback as tb
        send({"type": "error", "msg": str(exc), "trace": tb.format_exc()})
        # Give the SSE response time to reach the browser, then stop the server
        def _shutdown():
            time.sleep(1.5)
            _orig_stdout.write("\n[App] Stopping server due to pipeline error.\n")
            _orig_stdout.flush()
            os.kill(os.getpid(), signal.SIGTERM)
        threading.Thread(target=_shutdown, daemon=True).start()

    finally:
        _tl.write_fn = None
        try:
            os.unlink(audio_path)
        except OSError:
            pass
        job["status"] = "done"


# ── Embedded HTML page ───────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Transcription Engine</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:      #0d0d0f;
      --surface: #16161a;
      --surf2:   #1d1d24;
      --border:  #28283a;
      --text:    #e2e2ee;
      --muted:   #5a5a78;
      --accent:  #7c6af7;
      --blue:    #60a5fa;
      --green:   #4ade80;
      --red:     #f87171;
      --pink:    #f472b6;
      --r: 10px;
      --font: "Inter","Segoe UI",system-ui,sans-serif;
      --mono: "Cascadia Code","Fira Code","Consolas",monospace;
    }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      min-height: 100vh;
      padding: 2.5rem max(1.5rem, calc(50% - 44rem));
    }

    /* ── Header ────────────────────────────────────────────────────────── */
    header { text-align: center; margin-bottom: 2.5rem; }
    header h1 {
      font-size: 1.75rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #a78bfa, #60a5fa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 0.4rem;
    }
    header p {
      font-size: 0.73rem;
      color: var(--muted);
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }

    /* ── Drop zone ─────────────────────────────────────────────────────── */
    .dropzone {
      border: 2px dashed var(--border);
      border-radius: var(--r);
      padding: 3.5rem 2rem;
      text-align: center;
      cursor: pointer;
      background: var(--surface);
      transition: border-color 0.2s, background 0.2s;
      margin-bottom: 1.5rem;
      user-select: none;
    }
    .dropzone:hover, .dropzone.over {
      border-color: var(--accent);
      background: #14141e;
    }
    .dropzone.disabled { opacity: 0.45; pointer-events: none; }
    .dz-icon { font-size: 3rem; display: block; margin-bottom: 1rem; }
    .dropzone h2 { font-size: 1.05rem; font-weight: 600; margin-bottom: 0.4rem; }
    .dz-sub  { font-size: 0.74rem; color: var(--muted); }
    .dz-file {
      font-family: var(--mono);
      font-size: 0.78rem;
      color: var(--blue);
      margin-top: 0.6rem;
      display: block;
    }

    /* ── Progress bar ──────────────────────────────────────────────────── */
    .pbar-wrap {
      height: 3px;
      background: var(--border);
      border-radius: 2px;
      margin-bottom: 1.5rem;
      overflow: hidden;
    }
    .pbar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), var(--blue));
      border-radius: 2px;
      transition: width 0.4s ease;
    }

    /* ── Pipeline steps ────────────────────────────────────────────────── */
    .pipeline {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.75rem;
      margin-bottom: 1.25rem;
    }
    @media (max-width: 640px) { .pipeline { grid-template-columns: repeat(2, 1fr); } }

    .step {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r);
      padding: 0.9rem 1rem;
      display: flex;
      align-items: center;
      gap: 0.7rem;
      transition: border-color 0.3s;
    }
    .step.active { border-color: var(--blue); }
    .step.done   { border-color: var(--green); }
    .step.fail   { border-color: var(--red); }

    .dot {
      width: 9px; height: 9px;
      border-radius: 50%;
      background: var(--border);
      flex-shrink: 0;
      transition: background 0.3s;
    }
    .step.active .dot { background: var(--blue);  animation: pulse 1.1s infinite; }
    .step.done   .dot { background: var(--green); animation: none; }
    .step.fail   .dot { background: var(--red);   animation: none; }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50%       { opacity: 0.35; transform: scale(1.55); }
    }

    .step-name {
      font-size: 0.71rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .step.active .step-name,
    .step.done   .step-name { color: var(--text); }

    .step-stat { font-size: 0.67rem; color: var(--muted); margin-top: 0.2rem; }
    .step.active .step-stat { color: var(--blue);  }
    .step.done   .step-stat { color: var(--green); }
    .step.fail   .step-stat { color: var(--red);   }

    /* ── Log panel ─────────────────────────────────────────────────────── */
    .log-panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r);
      padding: 0.75rem 1rem;
      max-height: 190px;
      overflow-y: auto;
      margin-bottom: 1.5rem;
      font-family: var(--mono);
      font-size: 0.69rem;
      scroll-behavior: smooth;
    }
    .log-line { padding: 0.12rem 0; line-height: 1.55; color: #48486a; border-bottom: 1px solid #18181f; }
    .log-line.fresh { color: #8888aa; }
    .tag {
      font-size: 0.61rem;
      padding: 0.07rem 0.36rem;
      border-radius: 3px;
      margin-right: 0.45rem;
      font-weight: 700;
      text-transform: uppercase;
    }
    .t-load { background: #152640; color: #60a5fa; }
    .t-vad  { background: #1b1e3a; color: #818cf8; }
    .t-asr  { background: #11291a; color: #4ade80; }
    .t-hume { background: #2a1023; color: #f472b6; }

    /* ── Transcript output ─────────────────────────────────────────────── */
    .out-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
    }
    .out-title {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--muted);
    }
    .actions { display: flex; gap: 0.5rem; }
    .btn {
      background: var(--surf2);
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text);
      cursor: pointer;
      font-size: 0.72rem;
      padding: 0.38rem 0.88rem;
      transition: background 0.15s, border-color 0.15s;
      font-family: var(--font);
    }
    .btn:hover { background: #26263c; border-color: var(--accent); }

    /* ── Segments ──────────────────────────────────────────────────────── */
    .seg {
      display: grid;
      grid-template-columns: 9.5rem 1fr;
      gap: 1rem;
      padding: 0.7rem 1rem;
      border-left: 3px solid transparent;
      border-radius: 0 var(--r) var(--r) 0;
      margin-bottom: 3px;
      background: var(--surface);
      animation: fadeUp 0.3s ease both;
      transition: background 0.2s, border-color 0.6s;
    }
    .seg:hover { background: var(--surf2); }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(8px); }
      to   { opacity: 1; transform: none; }
    }
    .seg-meta { display: flex; flex-direction: column; gap: 0.25rem; }
    .seg-ts  { font-family: var(--mono); font-size: 0.64rem; color: var(--muted); }
    .seg-emo {
      font-size: 0.67rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: lowercase;
      opacity: 0;
      transition: opacity 0.6s, color 0.5s;
    }
    .seg-emo.show { opacity: 1; }
    .seg-text {
      font-size: 0.93rem;
      line-height: 1.75;
      color: var(--muted);
      align-self: center;
      transition: color 0.6s;
    }

    /* ── Error box ─────────────────────────────────────────────────────── */
    .errbox {
      background: #180a0a;
      border: 1px solid #5c1313;
      border-radius: var(--r);
      padding: 1rem 1.25rem;
      color: var(--red);
      font-family: var(--mono);
      font-size: 0.7rem;
      line-height: 1.7;
      white-space: pre-wrap;
      margin-bottom: 1.5rem;
    }

    .hidden { display: none !important; }
    #reset-btn { display: block; margin: 1.5rem auto 0; }
  </style>
</head>
<body>

<header>
  <h1>Emotion-Aware Transcription</h1>
  <p>pyannote VAD &nbsp;·&nbsp; CrisperWhisper &nbsp;·&nbsp; Hume AI Prosody</p>
</header>

<!-- Drop zone -->
<div id="dz" class="dropzone" onclick="document.getElementById('fi').click()">
  <span class="dz-icon">🎙</span>
  <h2>Drop audio here or click to select</h2>
  <p class="dz-sub">WAV &nbsp;·&nbsp; MP3 &nbsp;·&nbsp; M4A &nbsp;·&nbsp; FLAC &nbsp;·&nbsp; OGG</p>
  <span class="dz-file" id="fn"></span>
  <input type="file" id="fi" accept="audio/*" hidden>
</div>

<!-- Progress bar -->
<div class="pbar-wrap hidden" id="pw"><div class="pbar" id="pb"></div></div>

<!-- Pipeline step indicators -->
<div class="pipeline hidden" id="pipe">
  <div class="step" id="s-load">
    <div class="dot"></div>
    <div>
      <div class="step-name">Load Audio</div>
      <div class="step-stat">Waiting</div>
    </div>
  </div>
  <div class="step" id="s-vad">
    <div class="dot"></div>
    <div>
      <div class="step-name">Voice Activity</div>
      <div class="step-stat">Waiting</div>
    </div>
  </div>
  <div class="step" id="s-asr">
    <div class="dot"></div>
    <div>
      <div class="step-name">Transcription</div>
      <div class="step-stat">Waiting</div>
    </div>
  </div>
  <div class="step" id="s-hume">
    <div class="dot"></div>
    <div>
      <div class="step-name">Emotion Analysis</div>
      <div class="step-stat">Waiting</div>
    </div>
  </div>
</div>

<!-- Live log console -->
<div class="log-panel hidden" id="log"><div id="lc"></div></div>

<!-- Error display -->
<div class="errbox hidden" id="eb"></div>

<!-- Transcript output -->
<div id="out" class="hidden">
  <div class="out-header">
    <span class="out-title">Transcript</span>
    <div class="actions">
      <button class="btn" onclick="dlHTML()">Download HTML</button>
      <button class="btn" onclick="dlJSON()">Download JSON</button>
    </div>
  </div>
  <div id="segs"></div>
</div>

<button class="btn hidden" id="reset-btn" onclick="resetUI()">Transcribe Another File</button>

<script>
let jobId = null, resultSegs = null, origName = "";

const dz   = document.getElementById('dz');
const fi   = document.getElementById('fi');
const fn   = document.getElementById('fn');
const pw   = document.getElementById('pw');
const pb   = document.getElementById('pb');
const pipe = document.getElementById('pipe');
const log  = document.getElementById('log');
const lc   = document.getElementById('lc');
const out  = document.getElementById('out');
const segs = document.getElementById('segs');
const eb   = document.getElementById('eb');
const rstn = document.getElementById('reset-btn');

const STEP_ORDER = { load: 0, vad: 1, asr: 2, hume: 3 };

// ── Drag & drop ──────────────────────────────────────────────────────────────
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('over'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});
fi.addEventListener('change', () => { if (fi.files[0]) handleFile(fi.files[0]); });

// ── File handling ────────────────────────────────────────────────────────────
function handleFile(f) {
  origName = f.name;
  fn.textContent = f.name;
  dz.classList.add('disabled');
  pw.classList.remove('hidden');
  pipe.classList.remove('hidden');
  log.classList.remove('hidden');
  eb.classList.add('hidden');
  segs.innerHTML = '';
  out.classList.add('hidden');
  rstn.classList.add('hidden');
  resetSteps();

  const form = new FormData();
  form.append('file', f);

  fetch('/transcribe', { method: 'POST', body: form })
    .then(r => r.ok ? r.json() : r.json().then(j => { throw new Error(j.detail || r.statusText); }))
    .then(({ job_id }) => { jobId = job_id; listenSSE(job_id); })
    .catch(e => showErr(e.message));
}

// ── SSE stream ───────────────────────────────────────────────────────────────
function listenSSE(id) {
  const es = new EventSource('/stream/' + id);

  es.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.type === 'ping') return;

    if (m.type === 'step_start') {
      setStep(m.step, 'active');
      setProgress(STEP_ORDER[m.step], 4);
    }
    if (m.type === 'step_done') {
      setStep(m.step, 'done');
      setProgress(STEP_ORDER[m.step] + 1, 4);
    }
    if (m.type === 'log') addLog(m.step, m.msg);
    if (m.type === 'segment_transcribed') {
      out.classList.remove('hidden');
      addSeg(m);
    }
    if (m.type === 'done') {
      resultSegs = m.segments;
      pb.style.width = '100%';
      applyEmotions(m.segments);
      rstn.classList.remove('hidden');
      es.close();
    }
    if (m.type === 'error') {
      showErr(m.msg + (m.trace ? '\n\n' + m.trace : ''));
      rstn.classList.remove('hidden');
      es.close();
    }
  };

  es.onerror = () => {
    if (es.readyState !== EventSource.CLOSED) {
      showErr('Connection to server lost.');
      es.close();
    }
  };
}

// ── UI helpers ───────────────────────────────────────────────────────────────
function setStep(id, state) {
  const el = document.getElementById('s-' + id);
  if (!el) return;
  el.className = 'step ' + state;
  const stat = el.querySelector('.step-stat');
  stat.textContent = state === 'active' ? 'Running…' : state === 'done' ? 'Done' : 'Failed';
}

function setProgress(done, total) {
  pb.style.width = (done / total * 100) + '%';
}

function addLog(step, msg) {
  lc.querySelectorAll('.fresh').forEach(e => e.classList.remove('fresh'));
  const d = document.createElement('div');
  d.className = 'log-line fresh';
  const t = document.createElement('span');
  t.className = 'tag t-' + step;
  t.textContent = step.toUpperCase();
  d.appendChild(t);
  d.appendChild(document.createTextNode(msg));
  lc.appendChild(d);
  log.scrollTop = log.scrollHeight;
}

function addSeg(s) {
  const d = document.createElement('div');
  d.className = 'seg';
  d.id = 'sg' + s.n;
  d.style.animationDelay = (s.n * 0.04) + 's';
  d.innerHTML =
    '<div class="seg-meta">' +
      '<div class="seg-ts">' + fmtTs(s.start) + ' &rarr; ' + fmtTs(s.end) + '</div>' +
      '<div class="seg-emo" id="em' + s.n + '">&mdash;</div>' +
    '</div>' +
    '<div class="seg-text" id="st' + s.n + '">' + esc(s.text) + '</div>';
  segs.appendChild(d);
  d.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function applyEmotions(arr) {
  for (const s of arr) {
    const el = document.getElementById('sg' + s.n);
    if (!el) continue;
    el.style.borderLeftColor = s.color;
    const em = document.getElementById('em' + s.n);
    if (em) { em.textContent = s.emotion; em.style.color = s.color; em.classList.add('show'); }
    const st = document.getElementById('st' + s.n);
    if (st) st.style.color = s.color;
  }
}

function showErr(msg) {
  eb.textContent = msg;
  eb.classList.remove('hidden');
  document.querySelectorAll('.step.active').forEach(el => {
    el.className = 'step fail';
    el.querySelector('.step-stat').textContent = 'Failed';
  });
}

function fmtTs(sec) {
  const m = Math.floor(sec / 60);
  const s = String(Math.round(sec % 60)).padStart(2, '0');
  return String(m).padStart(2, '0') + ':' + s;
}

function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function resetSteps() {
  ['load', 'vad', 'asr', 'hume'].forEach(id => {
    const el = document.getElementById('s-' + id);
    el.className = 'step';
    el.querySelector('.step-stat').textContent = 'Waiting';
  });
  lc.innerHTML = '';
  pb.style.width = '0%';
}

// ── Downloads ────────────────────────────────────────────────────────────────
async function dlHTML() {
  if (!jobId) return;
  const r = await fetch('/result/' + jobId + '/html');
  if (!r.ok) return;
  _dl(await r.text(), origName.replace(/\.[^.]+$/, '') + '_transcript.html', 'text/html');
}

function dlJSON() {
  if (!resultSegs) return;
  _dl(
    JSON.stringify(resultSegs, null, 2),
    origName.replace(/\.[^.]+$/, '') + '_transcript.json',
    'application/json'
  );
}

function _dl(content, name, mime) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content], { type: mime }));
  a.download = name;
  a.click();
}

// ── Reset ────────────────────────────────────────────────────────────────────
function resetUI() {
  dz.classList.remove('disabled');
  fn.textContent = '';
  fi.value = '';
  pw.classList.add('hidden');
  pipe.classList.add('hidden');
  log.classList.add('hidden');
  out.classList.add('hidden');
  eb.classList.add('hidden');
  rstn.classList.add('hidden');
  segs.innerHTML = '';
  resetSteps();
  jobId = null;
  resultSegs = null;
}
</script>
</body>
</html>"""


# ── FastAPI routes ───────────────────────────────────────────────────────────
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML_PAGE)


@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    hume_key = os.environ.get("HUME_API_KEY", "")
    if not hume_key:
        raise HTTPException(status_code=400, detail="HUME_API_KEY not set in .env")
    if not os.environ.get("HF_TOKEN"):
        raise HTTPException(status_code=400, detail="HF_TOKEN not set in .env")

    suffix = Path(file.filename or "audio").suffix or ".tmp"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"q": queue.Queue(), "status": "running", "result": None}

    threading.Thread(
        target=_worker, args=(job_id, tmp.name, hume_key), daemon=True
    ).start()

    return JSONResponse({"job_id": job_id})


@app.get("/stream/{job_id}")
async def stream_events(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    q = _jobs[job_id]["q"]

    def generate():
        while True:
            try:
                msg = q.get(timeout=25)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/result/{job_id}/html", response_class=HTMLResponse)
async def result_html(job_id: str):
    job = _jobs.get(job_id)
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Result not ready")
    return HTMLResponse(job["result"]["html"])


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = 7860
    print(f"\n  Transcription Engine  →  http://localhost:{port}\n")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
