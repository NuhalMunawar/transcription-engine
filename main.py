#!/usr/bin/env python3
"""
Emotion-Aware Verbatim Audio Transcription Pipeline
====================================================
Accuracy is the highest priority.

Orchestration:
  Step 1 — pyannote/voice-activity-detection  (accurate VAD + segmentation)
  Step 2 — nyrahealth/CrisperWhisper          (verbatim ASR, beam search)
  Step 3 — Hume AI Batch API (prosody)        (per-segment emotion analysis)

Output:
  • Coloured terminal transcript (ANSI)
  • Self-contained HTML file  (<stem>_transcript.html)
  • Machine-readable JSON     (<stem>_transcript.json)

Usage:
    python main.py audio.wav
    python main.py audio.mp3 --output-dir results/
    HUME_API_KEY=xxx python main.py speech.m4a
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import html as html_lib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Third-party — imported lazily inside functions so missing extras give clear
# error messages rather than a stack trace on import.
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np

# torchaudio >= 2.5 removed list_audio_backends(); pyannote.audio 3.x still
# calls it at import time. Restore it as a no-op shim so the import succeeds.
try:
    import torchaudio as _ta
    if not hasattr(_ta, "list_audio_backends"):
        _ta.list_audio_backends = lambda: ["soundfile"]
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Constants / tunables
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE: int = 16_000          # Hz — all audio normalised to this

# Step 1 — segmentation
SILENCE_THRESHOLD: float = 3.0     # seconds: gap that starts a new segment
MIN_SEGMENT_DURATION: float = 0.3  # seconds: discard very short VAD hits

# Step 2 — transcription
ASR_MODEL_ID: str = "nyrahealth/CrisperWhisper"
ASR_NUM_BEAMS: int = 5             # beam search width; higher → more accurate

# Step 3 — emotion
EMOTION_TOP_N: int = 1             # how many emotions to surface per segment
HUME_POLL_INTERVAL: float = 4.0   # seconds between job-status polls

# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpeechSegment:
    start: float                       # seconds into original audio
    end: float
    audio: np.ndarray                  # float32 PCM @ SAMPLE_RATE
    raw_text: str = ""                 # CrisperWhisper raw output
    formatted_text: str = ""           # after formatting pass
    top_emotion: str = "neutral"
    emotion_score: float = 0.0
    emotion_color_hex: str = "#9E9E9E"
    unfinished: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Voice Activity Detection & Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def _load_audio(path: str) -> np.ndarray:
    """
    Load any audio file and return a float32 mono array at SAMPLE_RATE.
    Uses librosa for robust format support and high-quality resampling.
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required: pip install librosa")

    print(f"[Audio] Loading '{path}' …")
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    duration = len(audio) / SAMPLE_RATE
    print(f"[Audio] {duration:.1f}s  |  {SAMPLE_RATE} Hz  |  mono float32")
    return audio


def _run_pyannote_vad(audio: np.ndarray) -> List[Tuple[float, float]]:
    """
    Use pyannote/voice-activity-detection (SOTA accuracy) to get speech spans.
    Returns a list of (start_sec, end_sec) tuples, sorted chronologically.

    Requires a Hugging Face token with access to the gated pyannote models.
    Set HF_TOKEN environment variable before running.
    """
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        raise ImportError(
            "pyannote.audio is required: pip install pyannote.audio"
        )

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print(
            "[VAD] WARNING: HF_TOKEN not set. pyannote gated models may fail.\n"
            "          Visit https://hf.co/pyannote/voice-activity-detection "
            "and accept the terms, then set HF_TOKEN=<your_token>."
        )

    print("[VAD] Loading pyannote/voice-activity-detection …")
    pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection",
        use_auth_token=hf_token or None,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    print(f"[VAD] Running on {device} …")

    # pyannote accepts a dict with waveform (2-D tensor) + sample_rate
    import torch
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
    annotation = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})

    spans: List[Tuple[float, float]] = []
    for segment in annotation.get_timeline().support():
        if (segment.end - segment.start) >= MIN_SEGMENT_DURATION:
            spans.append((segment.start, segment.end))

    spans.sort(key=lambda x: x[0])
    print(f"[VAD] Found {len(spans)} speech spans")
    # Explicitly free the large VAD pipeline before returning
    del pipeline, annotation, waveform
    return spans


def build_segments(
    spans: List[Tuple[float, float]], audio: np.ndarray
) -> List[SpeechSegment]:
    """
    Merge consecutive VAD spans into SpeechSegments using the 3-second rule:
      - gap < SILENCE_THRESHOLD  → extend current segment
      - gap ≥ SILENCE_THRESHOLD  → start new segment
    """
    if not spans:
        return []

    segments: List[SpeechSegment] = []
    cur_start, cur_end = spans[0]

    for start, end in spans[1:]:
        gap = start - cur_end
        if gap < SILENCE_THRESHOLD:
            cur_end = end            # absorb into current segment
        else:
            s = int(cur_start * SAMPLE_RATE)
            e = int(cur_end   * SAMPLE_RATE)
            segments.append(
                SpeechSegment(start=cur_start, end=cur_end, audio=audio[s:e])
            )
            cur_start, cur_end = start, end

    # flush final segment
    s = int(cur_start * SAMPLE_RATE)
    e = int(cur_end   * SAMPLE_RATE)
    segments.append(
        SpeechSegment(start=cur_start, end=cur_end, audio=audio[s:e])
    )

    print(f"[VAD] Built {len(segments)} segment(s)  "
          f"(silence threshold = {SILENCE_THRESHOLD}s)")
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2A — CrisperWhisper Transcription
# ─────────────────────────────────────────────────────────────────────────────

def _load_asr_pipeline():
    """
    Load CrisperWhisper through HuggingFace Transformers.

    Accuracy settings:
      • float16 on GPU, float32 on CPU (no quantisation — avoids WER degradation)
      • beam search (ASR_NUM_BEAMS) instead of greedy
      • word-level timestamps for accurate Hume alignment downstream
    """
    try:
        import torch
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline as hf_pipeline,
        )
    except ImportError:
        raise ImportError(
            "transformers is required: pip install transformers accelerate safetensors"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    print(f"[ASR] Loading {ASR_MODEL_ID} on {device} …")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)

    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=device,
        generate_kwargs={
            "num_beams": ASR_NUM_BEAMS,   # beam search for accuracy
            "language":  "english",
            "task":      "transcribe",
        },
    )
    print("[ASR] Model loaded.")
    return pipe


def transcribe_segment(pipe, segment: SpeechSegment) -> str:
    """
    Transcribe a single SpeechSegment with word-level timestamps.
    Returns the raw verbatim text string.
    """
    result = pipe(
        {"array": segment.audio, "sampling_rate": SAMPLE_RATE},
    )
    return result["text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2B — Formatting Pass
# ─────────────────────────────────────────────────────────────────────────────

# Filler words that must always be wrapped in [brackets].
# Ordered longest-first so overlapping patterns match correctly.
_FILLERS = (
    "uh-huh", "mm-hmm", "mm-hm", "mhm",
    "uhh", "umm", "hmm", "hm", "uh", "um",
    "er", "err", "ah", "ahh", "mm",
)
_FILLER_RE = re.compile(
    r'(?<!\[)\b(' + "|".join(re.escape(f) for f in _FILLERS) + r')\b(?!\])',
    re.IGNORECASE,
)

# Already-bracketed token — e.g. [uh] or [uh-huh]
_BRACKETED_RE = re.compile(r'\[([^\]]+)\]')

# Stutter spacing artefacts: "I - I" or "m- my" → "I-I" / "m-my"
_STUTTER_SPACE_RE = re.compile(r'(\w)\s*-\s+(\w)')

# Trailing punctuation that indicates a finished thought
_TERMINAL_PUNCT = frozenset(".?!…")


def _protect_brackets(text: str) -> Tuple[str, Dict[str, str]]:
    """Replace existing [tokens] with placeholders so we don't double-wrap."""
    store: Dict[str, str] = {}
    counter = [0]

    def replace(m: re.Match) -> str:
        key = f"\x00PROT{counter[0]}\x00"
        counter[0] += 1
        inner = m.group(1).lower()
        inner = re.sub(r'[^\w\-]', '', inner)   # remove illegal punctuation
        store[key] = f"[{inner}]"
        return key

    return _BRACKETED_RE.sub(replace, text), store


def _restore_brackets(text: str, store: Dict[str, str]) -> str:
    for key, val in store.items():
        text = text.replace(key, val)
    return text


def formatting_pass(raw_text: str) -> str:
    """
    Enforce verbatim formatting rules on CrisperWhisper output:

    Rules applied in order:
      1. Protect already-bracketed tokens; strip illegal punctuation inside.
      2. Stutter notation: collapse "m- my" → "m-my" (no space around hyphen).
         Stutters are NEVER wrapped in brackets — they stay as plain text.
      3. Filler words (uh, um, hmm, …) → [filler].  No punctuation inside
         brackets except hyphens.
      4. Restore protected tokens.
    """
    # 1. Protect existing brackets
    text, protected = _protect_brackets(raw_text)

    # 2. Normalise stutter spacing: "I - I" or "I -I" → "I-I"
    text = _STUTTER_SPACE_RE.sub(r'\1-\2', text)

    # 3. Wrap bare filler words
    def _wrap_filler(m: re.Match) -> str:
        word = m.group(1).lower()
        word = re.sub(r'[^\w\-]', '', word)   # only letters, digits, hyphens
        return f"[{word}]"

    text = _FILLER_RE.sub(_wrap_filler, text)

    # 4. Restore protected tokens
    text = _restore_brackets(text, protected)

    return text.strip()


def mark_unfinished(segment: SpeechSegment) -> None:
    """
    If the segment's formatted text doesn't end with terminal punctuation,
    treat it as an interrupted or incomplete utterance and append '...'.
    """
    text = segment.formatted_text.rstrip()
    if text and text[-1] not in _TERMINAL_PUNCT:
        segment.formatted_text = text + "..."
        segment.unfinished = True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Hume AI Emotion Analysis
# ─────────────────────────────────────────────────────────────────────────────

# Full 48-emotion Hume palette → (ANSI escape code, HTML hex colour).
# Design rationale:
#   High-arousal positive  → warm yellows / oranges
#   Low-arousal positive   → greens
#   High-arousal negative  → reds / magentas
#   Low-arousal negative   → blues
#   Cognitive / neutral    → cyans / greys
#   Laughter               → pale yellow  (intentionally ≠ joy)
EMOTION_PALETTE: Dict[str, Tuple[str, str]] = {
    # ── High-arousal positive ──────────────────────────────────────────────
    "excitement":          ("\033[93m", "#FFC107"),
    "joy":                 ("\033[33m", "#FFD54F"),
    "elation":             ("\033[93m", "#FFD740"),
    "enthusiasm":          ("\033[93m", "#FFE082"),
    "pride":               ("\033[93m", "#FFB300"),
    "triumph":             ("\033[93m", "#FFA000"),
    "ecstasy":             ("\033[93m", "#FF8F00"),

    # ── Low-arousal positive ───────────────────────────────────────────────
    "amusement":           ("\033[92m", "#AED581"),   # ← NOT the same as joy
    "contentment":         ("\033[92m", "#81C784"),
    "satisfaction":        ("\033[32m", "#66BB6A"),
    "relief":              ("\033[92m", "#A5D6A7"),
    "calmness":            ("\033[92m", "#C8E6C9"),
    "serenity":            ("\033[92m", "#DCEDC8"),
    "love":                ("\033[31m", "#EF9A9A"),
    "adoration":           ("\033[31m", "#F48FB1"),
    "affection":           ("\033[91m", "#F06292"),
    "gratitude":           ("\033[32m", "#4CAF50"),
    "admiration":          ("\033[33m", "#FFE0B2"),
    "aesthetic_appreciation": ("\033[33m", "#FFCC80"),
    "interest":            ("\033[36m", "#4DD0E1"),
    "curiosity":           ("\033[96m", "#80DEEA"),
    "desire":              ("\033[91m", "#EF5350"),

    # ── Laughter / vocal bursts — explicitly NOT mapped to joy ────────────
    "laughter":            ("\033[33m", "#FFF176"),   # pale yellow, distinct

    # ── High-arousal negative ──────────────────────────────────────────────
    "anger":               ("\033[91m", "#EF5350"),
    "frustration":         ("\033[31m", "#E53935"),
    "disgust":             ("\033[35m", "#AB47BC"),
    "contempt":            ("\033[35m", "#8E24AA"),
    "horror":              ("\033[31m", "#B71C1C"),
    "distress":            ("\033[91m", "#EF9A9A"),
    "fear":                ("\033[95m", "#CE93D8"),
    "anxiety":             ("\033[35m", "#BA68C8"),
    "panic":               ("\033[91m", "#D32F2F"),

    # ── Low-arousal negative ───────────────────────────────────────────────
    "sadness":             ("\033[34m", "#42A5F5"),
    "grief":               ("\033[94m", "#90CAF9"),
    "disappointment":      ("\033[34m", "#64B5F6"),
    "loneliness":          ("\033[94m", "#BBDEFB"),
    "guilt":               ("\033[31m", "#EF9A9A"),
    "shame":               ("\033[31m", "#8D6E63"),
    "regret":              ("\033[34m", "#5C6BC0"),
    "melancholy":          ("\033[34m", "#7986CB"),
    "nostalgia":           ("\033[94m", "#9FA8DA"),
    "envy":                ("\033[32m", "#388E3C"),
    "boredom":             ("\033[37m", "#90A4AE"),

    # ── Cognitive / mixed ──────────────────────────────────────────────────
    "surprise":            ("\033[36m", "#26C6DA"),
    "confusion":           ("\033[96m", "#80DEEA"),
    "concentration":       ("\033[96m", "#B2EBF2"),
    "contemplation":       ("\033[96m", "#E0F7FA"),
    "realization":         ("\033[36m", "#00BCD4"),
    "awkwardness":         ("\033[37m", "#B0BEC5"),
    "sympathy":            ("\033[34m", "#64B5F6"),
    "empathic_pain":       ("\033[94m", "#7986CB"),
    "tiredness":           ("\033[37m", "#9E9E9E"),

    # ── Neutral / default ──────────────────────────────────────────────────
    "neutral":             ("\033[37m", "#BDBDBD"),
}
_DEFAULT_PALETTE: Tuple[str, str] = ("\033[37m", "#9E9E9E")


def _palette(emotion: str) -> Tuple[str, str]:
    """Return (ANSI escape, hex) for an emotion name, case-insensitive."""
    return EMOTION_PALETTE.get(emotion.lower(), _DEFAULT_PALETTE)


def analyse_emotions_hume(
    segments: List[SpeechSegment],
    audio_path: str,
    api_key: str,
) -> None:
    """
    Submit the full audio file to the Hume AI Batch Expression Measurement API
    with the Prosody model.  Uses hume SDK >= 0.7 (HumeClient).

    Each Hume prediction window has a time range and emotion scores.  We map
    those windows back to our segments using time-overlap weighting so that a
    segment that spans multiple Hume windows gets a properly averaged score
    rather than just the first window's value.

    NOTE on laughter:
      Hume has an explicit 'Laughter' emotion category.  We do NOT remap it
      to 'Joy' — it is displayed as its own category with a distinct colour.
    """
    try:
        from hume import HumeClient
        from hume.expression_measurement.batch.types import (
            InferenceBaseRequest,
            Models,
            Prosody,
        )
    except ImportError:
        raise ImportError("hume is required: pip install hume>=0.7.0")

    print("[Hume] Submitting audio for prosody/emotion analysis …")
    client = HumeClient(api_key=api_key)

    with open(audio_path, "rb") as audio_file:
        job_id: str = client.expression_measurement.batch.start_inference_job_from_local_file(
            file=[audio_file],
            json=InferenceBaseRequest(models=Models(prosody=Prosody())),
        )
    print(f"[Hume] Job ID: {job_id}  — polling every {HUME_POLL_INTERVAL}s …")

    # Poll until terminal state
    start_t = time.time()
    while True:
        details = client.expression_measurement.batch.get_job_details(id=job_id)
        state_status: str = details.state.status
        if state_status in ("COMPLETED", "FAILED"):
            break
        elapsed = int(time.time() - start_t)
        print(f"[Hume] Waiting … {elapsed}s elapsed", end="\r", flush=True)
        time.sleep(HUME_POLL_INTERVAL)

    if state_status == "FAILED":
        print("\n[Hume] WARNING: Hume job failed. Emotion tags will be skipped.")
        return
    print(f"\n[Hume] Job complete in {time.time() - start_t:.1f}s")

    raw_preds = client.expression_measurement.batch.get_job_predictions(id=job_id)

    # ── Parse Hume response (typed SDK objects) ───────────────────────────
    # raw_preds: List[InferenceSourcePredictResult]
    # Each  → .results.predictions[i].models.prosody.grouped_predictions[j].predictions
    windows: List[Tuple[float, float, List[Tuple[str, float]]]] = []
    try:
        for source_result in raw_preds:
            if source_result.results is None:
                continue
            for inference_pred in source_result.results.predictions:
                prosody_model = inference_pred.models.prosody
                if prosody_model is None:
                    continue
                for group in prosody_model.grouped_predictions:
                    for pred in group.predictions:
                        t_b = pred.time.begin
                        t_e = pred.time.end
                        emos = sorted(
                            [(e.name.lower(), e.score) for e in pred.emotions],
                            key=lambda x: x[1],
                            reverse=True,
                        )
                        windows.append((t_b, t_e, emos))
    except (AttributeError, TypeError) as exc:
        print(f"[Hume] WARNING: Could not parse response ({exc}). "
              "Emotion tags will be skipped.")
        return

    if not windows:
        print("[Hume] WARNING: No prosody windows found in response.")
        return

    # ── Map windows → segments via overlap-weighted averaging ────────────
    for seg in segments:
        score_acc: Dict[str, float] = {}
        total_overlap = 0.0

        for (wb, we, emos) in windows:
            overlap = max(0.0, min(seg.end, we) - max(seg.start, wb))
            if overlap <= 0.0:
                continue
            total_overlap += overlap
            for (name, score) in emos:
                score_acc[name] = score_acc.get(name, 0.0) + score * overlap

        if not score_acc or total_overlap == 0.0:
            seg.top_emotion   = "neutral"
            seg.emotion_score = 0.0
        else:
            # Normalise scores by total overlapping time
            normalised = {k: v / total_overlap for k, v in score_acc.items()}
            top_name, top_score = max(normalised.items(), key=lambda x: x[1])
            seg.top_emotion   = top_name
            seg.emotion_score = round(top_score, 4)

        _, hex_col            = _palette(seg.top_emotion)
        seg.emotion_color_hex = hex_col

    print("[Hume] Emotion analysis applied to all segments.")


# ─────────────────────────────────────────────────────────────────────────────
# Output Rendering
# ─────────────────────────────────────────────────────────────────────────────

_ANSI_RESET = "\033[0m"
_ANSI_DIM   = "\033[2m"
_ANSI_STRIP = re.compile(r'\033\[[0-9;]*m')


def _ts(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _ts_precise(seconds: float) -> str:
    """Format seconds as MM:SS.ss for exact display."""
    m, rem = divmod(seconds, 60)
    return f"{int(m):02d}:{rem:05.2f}"


def render_terminal(segments: List[SpeechSegment]) -> str:
    """
    Build the full ANSI-coloured terminal transcript string.
    Layout per segment:
        [MM:SS → MM:SS]  <emotion-in-colour>  text of the segment
    The emotion tag is always placed BEFORE the spoken text.
    """
    lines: List[str] = []
    for seg in segments:
        ansi, _ = _palette(seg.top_emotion)
        ts_str   = f"{_ANSI_DIM}[{_ts(seg.start)} → {_ts(seg.end)}]{_ANSI_RESET}"
        emo_str  = f"{ansi}{seg.top_emotion}{_ANSI_RESET}"
        lines.append(f"{ts_str}  {emo_str}  {seg.formatted_text}")
    return "\n\n".join(lines)


def render_html(segments: List[SpeechSegment], source_path: str) -> str:
    """
    Produce a polished, dark-themed, self-contained HTML transcript.
    Each row: segment number | timestamps | emotion-coloured text.
    """
    rows: List[str] = []
    for i, seg in enumerate(segments):
        seg_num = i + 1
        ts_str  = f"{_ts_precise(seg.start)} → {_ts_precise(seg.end)}"
        color   = seg.emotion_color_hex
        text    = html_lib.escape(seg.formatted_text)
        emotion = seg.top_emotion
        score   = f"{seg.emotion_score:.0%}"

        rows.append(
            f'<div class="segment" style="border-left-color:{color};">'
            f'<span class="seg-num">Segment {seg_num}</span>'
            f'<span class="ts">{ts_str}</span>'
            f'<span class="text" style="color:{color};" '
            f'title="{html_lib.escape(emotion)} · confidence {score}">'
            f'{text}'
            f'</span>'
            f'</div>'
        )

    source_name = Path(source_path).name
    rows_html   = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Transcript — {html_lib.escape(source_name)}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:      #0d0d0f;
      --surface: #16161a;
      --border:  #252530;
      --muted:   #555568;
      --radius:  8px;
      --font:    "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
      --mono:    "JetBrains Mono", "Fira Code", "Cascadia Code", monospace;
    }}

    body {{
      background: var(--bg);
      color: #e2e2e8;
      font-family: var(--font);
      font-size: 15px;
      line-height: 1.75;
      padding: 2.5rem max(2rem, calc(50% - 40rem));
      min-height: 100vh;
    }}

    header {{
      margin-bottom: 2.5rem;
      padding-bottom: 1.25rem;
      border-bottom: 1px solid var(--border);
    }}
    header h1 {{
      font-size: 1rem;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    header .filename {{
      font-family: var(--mono);
      font-size: 0.85rem;
      color: #777790;
      margin-top: 0.3rem;
    }}

    .segment {{
      display: grid;
      grid-template-columns: 7rem 13rem 1fr;
      align-items: baseline;
      gap: 0 1.25rem;
      padding: 0.7rem 1rem;
      border-left: 3px solid transparent;
      border-radius: 0 var(--radius) var(--radius) 0;
      margin-bottom: 3px;
      transition: background 0.12s;
    }}
    .segment:hover {{
      background: var(--surface);
    }}

    .seg-num {{
      font-family: var(--mono);
      font-size: 0.72rem;
      font-weight: 700;
      color: var(--muted);
      white-space: nowrap;
      user-select: none;
    }}

    .ts {{
      font-family: var(--mono);
      font-size: 0.72rem;
      color: var(--muted);
      white-space: nowrap;
      user-select: none;
    }}

    .text {{
      font-size: 0.97rem;
      word-break: break-word;
      /* color set inline per-segment */
    }}

    footer {{
      margin-top: 3rem;
      padding-top: 1.25rem;
      border-top: 1px solid var(--border);
      font-size: 0.75rem;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <header>
    <h1>Emotion-Aware Transcript</h1>
    <p class="filename">{html_lib.escape(source_name)}</p>
  </header>

  <main>
{rows_html}
  </main>

  <footer>
    Generated by the Emotion-Aware Transcription Pipeline
    &nbsp;·&nbsp; pyannote VAD &nbsp;·&nbsp; CrisperWhisper ASR
    &nbsp;·&nbsp; Hume AI Prosody
    &nbsp;·&nbsp; Hover over text to see emotion label
  </footer>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    audio_path: str,
    hume_api_key: str,
    output_dir: str = "output",
) -> None:
    sep = "─" * 62
    print(f"\n{sep}")
    print("  Emotion-Aware Transcription Pipeline")
    print(f"  Input : {audio_path}")
    print(f"{sep}\n")

    # ── Load audio ────────────────────────────────────────────────────────
    audio = _load_audio(audio_path)

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: VAD → Segments
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 1 — Voice Activity Detection")
    print(sep)
    spans    = _run_pyannote_vad(audio)
    segments = build_segments(spans, audio)

    if not segments:
        print("[Pipeline] No speech detected. Exiting.")
        return

    # Free VAD model from RAM before loading the much larger ASR model
    import gc
    gc.collect()
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass
    print("[Memory] VAD pipeline freed.")

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Transcription + Formatting
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 2 — CrisperWhisper Transcription + Formatting Pass")
    print(sep)
    pipe = _load_asr_pipeline()

    for i, seg in enumerate(segments):
        raw            = transcribe_segment(pipe, seg)
        seg.raw_text   = raw
        seg.formatted_text = formatting_pass(raw)
        mark_unfinished(seg)
        # Free audio buffer immediately after transcription to save RAM
        seg.audio = None
        print(f"  [{i+1:02d}/{len(segments):02d}]  "
              f"{_ts(seg.start)} → {_ts(seg.end)}  "
              f"({seg.duration:.1f}s){'  ✂' if seg.unfinished else ''}")
        print(f"          {seg.formatted_text}\n")

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Hume AI Emotion Analysis
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 3 — Hume AI Emotion Analysis")
    print(sep)
    analyse_emotions_hume(segments, audio_path, hume_api_key)

    # ─────────────────────────────────────────────────────────────────────
    # Render outputs
    # ─────────────────────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem    = Path(audio_path).stem

    # Terminal
    terminal_str = render_terminal(segments)
    print(f"\n{sep}")
    print("  FINAL TRANSCRIPT")
    print(f"{sep}\n")
    print(terminal_str)

    # HTML
    html_content = render_html(segments, audio_path)
    html_path    = out_dir / f"{stem}_transcript.html"
    html_path.write_text(html_content, encoding="utf-8")

    # Plain text (ANSI stripped)
    txt_path = out_dir / f"{stem}_transcript.txt"
    txt_path.write_text(
        _ANSI_STRIP.sub("", terminal_str), encoding="utf-8"
    )

    # JSON
    json_data = [
        {
            "segment":    i + 1,
            "start_sec":  seg.start,
            "end_sec":    seg.end,
            "duration_sec": round(seg.duration, 3),
            "emotion":    seg.top_emotion,
            "emotion_confidence": seg.emotion_score,
            "unfinished": seg.unfinished,
            "text":       seg.formatted_text,
            "raw_text":   seg.raw_text,
        }
        for i, seg in enumerate(segments)
    ]
    json_path = out_dir / f"{stem}_transcript.json"
    json_path.write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{sep}")
    print(f"  Outputs saved to: {out_dir.resolve()}")
    print(f"    • {html_path.name}   (visual, colour-coded)")
    print(f"    • {txt_path.name}   (plain text)")
    print(f"    • {json_path.name}  (structured data)")
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Emotion-aware verbatim audio transcription pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables
---------------------
  HUME_API_KEY   Hume AI API key (required)
  HF_TOKEN       Hugging Face token for pyannote gated models (required)

Examples
--------
  python main.py interview.wav
  python main.py podcast.mp3 --output-dir results/
  HUME_API_KEY=xxx HF_TOKEN=yyy python main.py speech.m4a
""",
    )
    p.add_argument("audio",
                   help="Path to the input audio file (wav, mp3, m4a, flac, …)")
    p.add_argument("--hume-api-key",
                   default=os.environ.get("HUME_API_KEY", ""),
                   metavar="KEY",
                   help="Hume AI API key (default: $HUME_API_KEY)")
    p.add_argument("--output-dir",
                   default="output",
                   metavar="DIR",
                   help="Directory for output files (default: ./output)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not Path(args.audio).exists():
        print(f"ERROR: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    if not args.hume_api_key:
        print(
            "ERROR: Hume AI API key is required.\n"
            "  Pass --hume-api-key <key>  or  export HUME_API_KEY=<key>",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.environ.get("HF_TOKEN"):
        print(
            "WARNING: HF_TOKEN is not set. pyannote gated models may fail.\n"
            "  Visit https://hf.co/pyannote/voice-activity-detection, "
            "accept the terms,\n  then: export HF_TOKEN=<your_token>\n",
            file=sys.stderr,
        )

    run_pipeline(
        audio_path   = args.audio,
        hume_api_key = args.hume_api_key,
        output_dir   = args.output_dir,
    )


if __name__ == "__main__":
    main()
