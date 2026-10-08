"""
Manages the persistent `asr_stream_server` subprocess (compiled from
github.com/microsoft/VibeASR.cpp) that performs the actual ASR inference
against the quantized VibeVoice-ASR-BitNet GGUF weights.

The model is not loaded via `transformers` — VibeVoice-ASR-BitNet ships as
two GGUF files (VAE encoder + BitNet-quantized LM decoder) meant to run
through this C++ engine. We spawn the engine once at startup, keep it
resident, and talk to it over stdin/stdout using its line-based protocol:
  - startup:        engine prints "---READY---" once models are loaded
  - per request:     write "<audio_path>\n" to stdin
  - per response:     read lines until "---END---" (or a "[ERROR] ..." line)
"""

import asyncio
import logging
import os
import subprocess

logger = logging.getLogger("vibeasr.loader")

ASR_SERVER_BIN = os.getenv("ASR_SERVER_BIN", "/app/bin/asr_stream_server")
VAE_MODEL_PATH = os.getenv("VAE_MODEL_PATH", "/app/models/vibeasr-vae-encoder-i8_s.gguf")
LM_MODEL_PATH = os.getenv("LM_MODEL_PATH", "/app/models/vibeasr-lm-i2_s-embed-q6_k.gguf")

ASR_THREADS = int(os.getenv("ASR_THREADS", "4"))
ASR_MAX_TOKENS = int(os.getenv("ASR_MAX_TOKENS", "4096"))
ASR_PROMPT_FORMAT = os.getenv("ASR_PROMPT_FORMAT", "text")
ASR_GREEDY = os.getenv("ASR_GREEDY", "true").lower() == "true"
ASR_TEMPERATURE = float(os.getenv("ASR_TEMPERATURE", "0.7"))
ASR_TOP_P = float(os.getenv("ASR_TOP_P", "0.9"))
ASR_CONTEXT = os.getenv("ASR_CONTEXT", "")

STARTUP_TIMEOUT_SEC = float(os.getenv("ASR_STARTUP_TIMEOUT", "120"))
SHUTDOWN_TIMEOUT_SEC = 5.0

MODEL_NAME = "VibeVoice-ASR-BitNet"


class ASREngineError(RuntimeError):
    pass


class ASREngine:
    """Owns one `asr_stream_server` subprocess. Not safe for concurrent
    callers without external synchronization — callers must hold `lock`."""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.lock = asyncio.Lock()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self):
        if self.is_running():
            return

        for path, label in ((ASR_SERVER_BIN, "ASR_SERVER_BIN"),
                             (VAE_MODEL_PATH, "VAE_MODEL_PATH"),
                             (LM_MODEL_PATH, "LM_MODEL_PATH")):
            if not os.path.exists(path):
                raise ASREngineError(f"{label} not found at {path}")

        cmd = [
            ASR_SERVER_BIN,
            "--vae-model", VAE_MODEL_PATH,
            "--lm-model", LM_MODEL_PATH,
            "-t", str(ASR_THREADS),
            "--max-tokens", str(ASR_MAX_TOKENS),
            "--prompt-format", ASR_PROMPT_FORMAT,
        ]
        if ASR_GREEDY:
            cmd.append("--greedy")
        else:
            cmd.extend(["--temperature", str(ASR_TEMPERATURE), "--top-p", str(ASR_TOP_P)])
        if ASR_CONTEXT:
            cmd.extend(["--context", ASR_CONTEXT])

        logger.info("Starting asr_stream_server: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )

        ready_line = self.process.stdout.readline().decode(errors="replace").strip()
        if ready_line != "---READY---":
            stderr_tail = self.process.stderr.read(4096).decode(errors="replace")
            self.stop()
            raise ASREngineError(
                f"asr_stream_server failed to become ready (got {ready_line!r}): {stderr_tail}"
            )
        logger.info("asr_stream_server ready (pid=%s)", self.process.pid)

    def stop(self):
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                self.process.stdin.write(b"EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=SHUTDOWN_TIMEOUT_SEC)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        finally:
            self.process = None

    def set_context(self, hotwords: str):
        if not self.is_running():
            return
        self.process.stdin.write(f"CONTEXT:{hotwords}\n".encode())
        self.process.stdin.flush()
        self.process.stdout.readline()  # engine acks with one line

    def transcribe_sync(self, audio_path: str, hotwords: str | None = None) -> str:
        """Blocking. Caller must hold `lock` (via run_in_executor) so only
        one request talks to the subprocess's stdin/stdout at a time."""
        if not self.is_running():
            self.start()

        if hotwords:
            self.set_context(hotwords)

        self.process.stdin.write(f"{audio_path}\n".encode())
        self.process.stdin.flush()

        chunks = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                stderr_tail = self.process.stderr.read(4096).decode(errors="replace")
                raise ASREngineError(f"asr_stream_server exited unexpectedly: {stderr_tail}")
            text = line.decode("utf-8", errors="replace")
            if text.strip() == "---END---":
                break
            if text.startswith("[ERROR]"):
                raise ASREngineError(text.strip())
            chunks.append(text.rstrip("\n"))
        return "".join(chunks)


_engine = ASREngine()


def get_engine() -> ASREngine:
    return _engine
