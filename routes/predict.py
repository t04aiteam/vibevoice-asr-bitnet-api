import asyncio
import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from middleware.auth import verify_key
from models.loader import MODEL_NAME, ASREngineError, get_engine
from schemas.request import TranscribeResponse

router = APIRouter()

SUPPORTED_EXTENSIONS = {".wav", ".mp3"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))  # 200 MB


@router.post("/transcribe", response_model=TranscribeResponse, dependencies=[Depends(verify_key)])
async def transcribe(
    audio: UploadFile = File(..., description="WAV or MP3 audio file to transcribe"),
    hotwords: str | None = Form(None, description="Optional comma/newline separated hotwords for improved recognition"),
):
    """Transcribe an uploaded audio file to text using VibeVoice-ASR-BitNet."""
    ext = os.path.splitext(audio.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    data = await audio.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds max size of {MAX_UPLOAD_BYTES} bytes")
    if not data:
        raise HTTPException(400, "Uploaded file is empty")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        engine = get_engine()
        loop = asyncio.get_event_loop()
        async with engine.lock:
            text = await loop.run_in_executor(None, engine.transcribe_sync, tmp_path, hotwords)

        return TranscribeResponse(text=text, model=MODEL_NAME)
    except ASREngineError as e:
        raise HTTPException(500, str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
