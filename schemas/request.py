from pydantic import BaseModel, Field


class TranscribeResponse(BaseModel):
    text: str = Field(..., description="Transcribed text")
    model: str = Field(..., description="Model used for transcription")


class HealthResponse(BaseModel):
    status: str
    model: str
    engine_running: bool
