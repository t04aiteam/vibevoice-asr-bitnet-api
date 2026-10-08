# VibeVoice-ASR-BitNet API

A FastAPI wrapper around [microsoft/VibeVoice-ASR-BitNet](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet),
a BitNet-quantized, CPU-only multilingual speech-to-text model (English, Chinese, French,
Italian, Korean, Portuguese, Vietnamese, and more).

**Important:** this model does not load through `transformers`. It ships as two GGUF files
(a VAE tokenizer + a BitNet/I2_S-quantized LM decoder) that run through a companion C++
inference engine, [microsoft/VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp). This API
compiles that engine, runs it as a persistent subprocess (`asr_stream_server`) started once at
app startup, and proxies transcription requests to it over its stdin/stdout protocol.

## Prerequisites

- Docker (recommended) — the build stage compiles C++ from source and needs `git`, `cmake`,
  and a C++ compiler. Build takes several minutes and ~2-3 GB of disk for build artifacts;
  the final runtime image only keeps the compiled binaries + ~1.6 GB of GGUF weights.
- No GPU required — this model is specifically built for real-time CPU inference (the engine
  uses AVX2/NEON SIMD kernels).
- If running outside Docker: Python ≥ 3.9, CMake ≥ 3.14, and a C++11-capable compiler
  (GCC/Clang on Linux/macOS; MinGW-w64 on Windows — not MSVC).

## Setup

### Option A — Docker (recommended)

```bash
cp .env.example .env   # set API_KEY at minimum
docker compose up --build
```

The build downloads and compiles [microsoft/VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp)
and pulls the GGUF weights from the model repo automatically — no manual model download needed.

### Option B — Run locally without Docker

```bash
# 1. Build the inference engine and fetch the GGUF weights
git clone --recursive https://github.com/microsoft/VibeASR.cpp.git
cd VibeASR.cpp
pip install huggingface_hub
python setup_env.py --hf-repo microsoft/VibeVoice-ASR-BitNet
cd ..

# 2. Point this API at the binaries and weights it just built
cp .env.example .env
# edit .env:
#   ASR_SERVER_BIN=./VibeASR.cpp/build/bin/asr_stream_server
#   VAE_MODEL_PATH=./VibeASR.cpp/models/vibeasr/vibeasr-vae-encoder-i8_s.gguf
#   LM_MODEL_PATH=./VibeASR.cpp/models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf

# 3. Install this API's dependencies and run it
pip install -r requirements.txt
uvicorn main:app --reload
```

## Environment Variables

| Variable | Required | Description | Example |
|---|---|---|---|
| `API_KEY` | Yes | API key clients must send in `X-Api-Key` | `changeme` |
| `ASR_SERVER_BIN` | No | Path to the compiled `asr_stream_server` binary | `/app/bin/asr_stream_server` |
| `VAE_MODEL_PATH` | No | Path to the VAE encoder GGUF file | `/app/models/vibeasr-vae-encoder-i8_s.gguf` |
| `LM_MODEL_PATH` | No | Path to the LM decoder GGUF file | `/app/models/vibeasr-lm-i2_s-embed-q6_k.gguf` |
| `ASR_THREADS` | No | CPU threads the engine uses per transcription | `4` |
| `ASR_MAX_TOKENS` | No | Max tokens the LM decoder can generate | `4096` |
| `ASR_PROMPT_FORMAT` | No | `text` or `json` output format from the engine | `text` |
| `ASR_GREEDY` | No | Use greedy decoding instead of sampling | `true` |
| `ASR_TEMPERATURE` | No | Sampling temperature (ignored if `ASR_GREEDY=true`) | `0.7` |
| `ASR_TOP_P` | No | Nucleus sampling top-p (ignored if `ASR_GREEDY=true`) | `0.9` |
| `ASR_CONTEXT` | No | Default hotwords/context to bias recognition | `` |
| `ASR_STARTUP_TIMEOUT` | No | Seconds to wait for the engine to report ready | `120` |
| `MAX_UPLOAD_BYTES` | No | Max accepted upload size in bytes | `209715200` |

## Running

- Locally: `uvicorn main:app --reload`
- Docker: `docker compose up --build`

## Endpoints

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/health` | — | `{"status": "ok", "model": str, "engine_running": bool}` |
| POST | `/api/v1/transcribe` | multipart form: `audio` (file, wav/mp3), `hotwords` (optional str) | `{"text": str, "model": str}` |

Both `/api/v1/*` endpoints require an `X-Api-Key` header.

### Example: health check

```bash
curl http://localhost:8000/health
```

### Example: transcribe a file

```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -H "X-Api-Key: changeme" \
  -F "audio=@sample.wav"
```

### Example: transcribe with hotwords for better recognition of names/jargon

```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -H "X-Api-Key: changeme" \
  -F "audio=@meeting.wav" \
  -F "hotwords=VibeVoice, Microsoft, BitNet"
```

## Notes & Limitations

- **Only one transcription runs at a time.** The engine is a single persistent subprocess
  communicating over stdin/stdout; concurrent requests are serialized behind an `asyncio.Lock`
  in [`models/loader.py`](models/loader.py) rather than run in parallel. This matches the
  model's design point (real-time on a handful of CPU threads per request), but if you need
  to serve many concurrent users, run multiple containers behind a load balancer rather than
  raising `--workers` on a single container (each worker would load its own copy of the model).
- **Supported audio formats are WAV and MP3** — the underlying engine decodes these via
  `dr_wav`/`dr_mp3` and resamples/normalizes internally, so you don't need to pre-process
  sample rate or channels client-side.
- **Streaming/incremental transcription** (as shown in the upstream Gradio demo, which
  simulates live audio with chunking + overlap + dedup) is not implemented here — this API
  wraps single-shot, whole-file transcription. It could be added as a `/stream` endpoint using
  the same `asr_stream_server` process if you need incremental output.
- Per the model card, accuracy is close to but slightly behind the full-size
  `VibeVoice-ASR-7B` model (see WER tables on the
  [model page](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet)) in exchange for
  ~2.9x compression and CPU-only real-time inference.
