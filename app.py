"""FastAPI application for zero-shot speech synthesis with F5-TTS."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

import soundfile as sf
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.engine import (
    EngineOutOfMemoryError,
    InvalidVoiceNameError,
    InvalidVoiceProfileError,
    SynthesisError,
    VoiceCloneEngine,
    VoiceProfileNotFoundError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)
ENGINE = VoiceCloneEngine()


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: str = Field(min_length=1, max_length=20_000)
    voice_name: str = Field(default="deep_male", min_length=1, max_length=64)
    speed: float = Field(default=0.9, gt=0.1, le=2.0)
    nfe_step: int = Field(default=32, ge=1, le=128)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class SynthesisResponse(BaseModel):
    status: str
    output_path: str
    duration_seconds: float


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


class VoicesResponse(BaseModel):
    voices: list[str]
    count: int


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    LOGGER.info("Starting voice clone service")
    await run_in_threadpool(ENGINE.load_model)
    yield
    LOGGER.info("Stopping voice clone service")


app = FastAPI(
    title="F5-TTS Voice Clone Service",
    version="1.0.0",
    description="Zero-shot voice cloning and text-to-speech microservice.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", device=ENGINE.device, model_loaded=ENGINE.is_loaded)


@app.get("/voices", response_model=VoicesResponse)
def voices() -> VoicesResponse:
    available_voices = ENGINE.list_voices()
    return VoicesResponse(voices=available_voices, count=len(available_voices))


@app.post(
    "/v1/tts/synthesize",
    response_model=None,
    responses={
        200: {"description": "Synthesis metadata or a WAV stream"},
        404: {"description": "Voice profile not found"},
        422: {"description": "Invalid input or voice profile"},
        500: {"description": "Inference failed"},
        503: {"description": "Device memory exhausted"},
    },
)
async def synthesize(
    request: SynthesisRequest,
    stream: Annotated[bool, Query(description="Return the generated WAV file directly")] = False,
) -> SynthesisResponse | FileResponse:
    try:
        output = await run_in_threadpool(
            ENGINE.synthesize,
            request.text,
            request.voice_name,
            request.speed,
            request.nfe_step,
        )
        output_path = Path(output)
        audio_info = sf.info(output_path)
        duration = float(audio_info.duration)

        if stream:
            return FileResponse(
                path=output_path,
                media_type="audio/wav",
                filename=output_path.name,
                headers={"X-Audio-Duration-Seconds": f"{duration:.3f}"},
            )

        return SynthesisResponse(
            status="success",
            output_path=str(output_path),
            duration_seconds=round(duration, 3),
        )
    except VoiceProfileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (InvalidVoiceNameError, InvalidVoiceProfileError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except EngineOutOfMemoryError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except SynthesisError as exc:
        LOGGER.error("Synthesis request failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Speech synthesis failed; check server logs for details",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
