import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.loader import MODEL_NAME, get_engine
from routes.predict import router
from schemas.request import HealthResponse

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_engine()
    engine.start()  # warm up at startup — fails fast if binaries/weights are missing
    yield
    engine.stop()


app = FastAPI(title="VibeVoice-ASR-BitNet API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse)
async def health():
    engine = get_engine()
    return HealthResponse(status="ok", model=MODEL_NAME, engine_running=engine.is_running())


@app.exception_handler(Exception)
async def global_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": str(exc)})
