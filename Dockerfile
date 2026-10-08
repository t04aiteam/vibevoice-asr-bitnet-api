# --- Stage 1: build the VibeASR.cpp inference engine + fetch GGUF weights ---
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
ARG ENGINE_REPO=https://github.com/microsoft/VibeASR.cpp.git
RUN git clone --recursive ${ENGINE_REPO} VibeASR.cpp
WORKDIR /build/VibeASR.cpp

RUN pip install --no-cache-dir huggingface_hub

# setup_env.py installs gguf-py from the llama.cpp submodule, builds the
# `asr_infer` / `asr_stream_server` binaries with cmake, and downloads the
# pre-quantized GGUF weights (safetensors/bin/pt are excluded — GGUF only).
ARG HF_REPO=microsoft/VibeVoice-ASR-BitNet
ARG BUILD_THREADS=4
RUN python3 setup_env.py --hf-repo ${HF_REPO} --threads ${BUILD_THREADS}

# --- Stage 2: slim runtime image with only the compiled binaries + API code ---
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/bin /app/models
COPY --from=builder /build/VibeASR.cpp/build/bin/asr_infer /app/bin/asr_infer
COPY --from=builder /build/VibeASR.cpp/build/bin/asr_stream_server /app/bin/asr_stream_server
COPY --from=builder /build/VibeASR.cpp/models/vibeasr/*.gguf /app/models/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
