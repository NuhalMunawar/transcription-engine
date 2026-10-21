# Emotion-Aware Transcription Engine

An audio transcription pipeline that combines voice activity detection, verbatim speech recognition, and emotion analysis. It can run from the command line or through a local browser interface.

## How it works

1. **Voice activity detection:** `pyannote.audio` detects speech and divides the recording into segments.
2. **Transcription:** CrisperWhisper transcribes each segment.
3. **Emotion analysis:** Hume AI's Prosody model analyzes the recording and associates emotion scores with the segments.
4. **Output:** The command-line pipeline prints a transcript and saves HTML, plain-text, and JSON files.

## Requirements

- Python and pip
- A Hugging Face account and read-access token
- A Hume AI API key
- Enough disk space for the CrisperWhisper model download (approximately 3.2 GB)
- A CUDA-capable GPU is recommended for faster transcription; CPU is also supported

The pipeline uses gated Hugging Face models. Before running it, accept the terms for [pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).

## Setup

Create and activate a virtual environment, then install the project requirements:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a CUDA-enabled PyTorch build, install the appropriate `torch` and `torchaudio` wheels for your CUDA version, following the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

Create a `.env` file in the project root with your credentials:

```dotenv
HF_TOKEN=your_hugging_face_read_token
HUME_API_KEY=your_hume_api_key
```

`.env` is ignored by Git. Do not commit API keys or access tokens.

## Command-line usage

Transcribe an audio file:

```powershell
python main.py path\to\audio.wav
```

Choose a different output directory:

```powershell
python main.py path\to\audio.mp3 --output-dir results
```

The `HUME_API_KEY` can also be supplied directly:

```powershell
python main.py path\to\audio.m4a --hume-api-key your_hume_api_key
```

The Hugging Face token is read from `HF_TOKEN`. The app supports common audio formats such as WAV, MP3, M4A, and FLAC. Model weights are downloaded from Hugging Face on first use and cached for later runs.

For an input file named `interview.wav`, the default `output/` directory receives:

- `interview_transcript.html` — formatted transcript
- `interview_transcript.txt` — plain-text transcript
- `interview_transcript.json` — structured segment data, including timing, text, and emotion scores

## Browser interface

Install the additional web dependencies and start the local interface:

```powershell
pip install fastapi "uvicorn[standard]"
python app.py
```

The app opens at [http://localhost:7860](http://localhost:7860). Upload an audio file in the browser to start transcription and follow progress as it runs. The interface requires both `HF_TOKEN` and `HUME_API_KEY` in `.env`.

## Optional model download helpers

To download CrisperWhisper ahead of time, with standard download output:

```powershell
python download_model.py
```

To use the browser-based download progress page instead, install the web dependencies above and run:

```powershell
python download_progress.py
```

The progress page runs at [http://localhost:7861](http://localhost:7861). Both helpers use the Hugging Face token from `.env`.

## Privacy

Audio is processed locally for voice activity detection and transcription. The full audio file is also submitted to Hume AI for prosody and emotion analysis. Review the relevant service terms and privacy policies before processing sensitive recordings.
