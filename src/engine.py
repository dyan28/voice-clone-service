"""F5-TTS voice-cloning engine with single-process model caching."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from .text_splitter import split_text

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VOICES_DIR = PROJECT_ROOT / "voices"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VOICE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class VoiceCloneError(RuntimeError):
    """Base exception for voice-cloning failures."""


class InvalidVoiceNameError(VoiceCloneError):
    """Raised when a voice name could escape the voices directory."""


class VoiceProfileNotFoundError(VoiceCloneError):
    """Raised when a voice profile is missing audio or transcript data."""


class InvalidVoiceProfileError(VoiceCloneError):
    """Raised when a voice profile cannot be read or has no transcript."""


class EngineOutOfMemoryError(VoiceCloneError):
    """Raised when the selected inference backend runs out of memory."""


class SynthesisError(VoiceCloneError):
    """Raised when F5-TTS inference fails."""


class VoiceCloneEngine:
    """Process-wide F5-TTS engine.

    Constructing this class repeatedly returns the same object. The model itself
    is loaded under a lock and therefore initialized at most once per process.
    FastAPI preloads it during application startup; CLI usage loads it on demand.
    """

    _instance: VoiceCloneEngine | None = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> VoiceCloneEngine:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self.device = self.detect_device()
        self.voices_dir = Path(os.getenv("VOICES_DIR", DEFAULT_VOICES_DIR)).expanduser().resolve()
        self.outputs_dir = Path(os.getenv("OUTPUTS_DIR", DEFAULT_OUTPUTS_DIR)).expanduser().resolve()
        self.model_name = os.getenv("F5_MODEL_NAME", "F5TTS_v1_Base")
        self.checkpoint_file = os.getenv("F5_CHECKPOINT_FILE", "").strip()
        self.vocab_file = os.getenv("F5_VOCAB_FILE", "").strip()
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()

        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("VoiceCloneEngine initialized with device=%s", self.device)

    @staticmethod
    def detect_device() -> str:
        """Choose MPS first, then CUDA, and finally CPU."""
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        """Load the F5-TTS model exactly once for this process."""
        if self._model is not None:
            return

        with self._model_lock:
            if self._model is not None:
                return

            started_at = time.perf_counter()
            try:
                from f5_tts.api import F5TTS

                options: dict[str, Any] = {
                    "model": self.model_name,
                    "device": self.device,
                }
                if self.checkpoint_file:
                    options["ckpt_file"] = self.checkpoint_file
                if self.vocab_file:
                    options["vocab_file"] = self.vocab_file

                LOGGER.info("Loading F5-TTS model=%s on device=%s", self.model_name, self.device)
                self._model = F5TTS(**options)
            except Exception as exc:
                if self._is_out_of_memory(exc):
                    self._release_device_cache()
                    raise EngineOutOfMemoryError(
                        f"Not enough {self.device} memory to load F5-TTS"
                    ) from exc
                raise SynthesisError(f"Unable to load F5-TTS: {exc}") from exc

            LOGGER.info("F5-TTS loaded in %.2f seconds", time.perf_counter() - started_at)

    def list_voices(self) -> list[str]:
        """Return profile names that have both WAV and TXT files."""
        wav_names = {path.stem for path in self.voices_dir.glob("*.wav") if path.is_file()}
        txt_names = {path.stem for path in self.voices_dir.glob("*.txt") if path.is_file()}
        return sorted(wav_names & txt_names)

    def synthesize(
        self,
        text: str,
        voice_name: str,
        speed: float = 0.9,
        nfe_step: int = 32,
    ) -> str:
        """Clone a voice and synthesize text into a WAV file.

        Returns the absolute path of the generated file.
        """
        normalized_text = " ".join(text.split())
        if not normalized_text:
            raise ValueError("text must not be empty")
        if speed <= 0:
            raise ValueError("speed must be greater than 0")
        if nfe_step < 1:
            raise ValueError("nfe_step must be at least 1")

        reference_audio, reference_text = self._resolve_voice_profile(voice_name)
        chunks = split_text(normalized_text, min_words=20, max_words=40)
        self.load_model()

        started_at = time.perf_counter()
        output_path = self.outputs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex}.wav"
        temporary_path = output_path.with_suffix(".tmp.wav")

        LOGGER.info(
            "Starting synthesis voice=%s chunks=%d speed=%.2f nfe_step=%d device=%s",
            voice_name,
            len(chunks),
            speed,
            nfe_step,
            self.device,
        )

        try:
            with self._inference_lock:
                audio = self._infer_chunks(
                    chunks=chunks,
                    reference_audio=reference_audio,
                    reference_text=reference_text,
                    speed=speed,
                    nfe_step=nfe_step,
                )
                waveform, sample_rate = audio
                sf.write(temporary_path, waveform, sample_rate, format="WAV", subtype="PCM_16")
                os.replace(temporary_path, output_path)
        except Exception as exc:
            temporary_path.unlink(missing_ok=True)
            if isinstance(exc, VoiceCloneError):
                raise
            if self._is_out_of_memory(exc):
                self._release_device_cache()
                raise EngineOutOfMemoryError(
                    f"Not enough {self.device} memory to synthesize audio"
                ) from exc
            LOGGER.exception("F5-TTS synthesis failed")
            raise SynthesisError(f"F5-TTS synthesis failed: {exc}") from exc

        elapsed = time.perf_counter() - started_at
        duration = len(waveform) / sample_rate
        LOGGER.info(
            "Synthesis completed path=%s duration=%.2fs inference_time=%.2fs",
            output_path,
            duration,
            elapsed,
        )
        return str(output_path.resolve())

    def _resolve_voice_profile(self, voice_name: str) -> tuple[Path, str]:
        if not VOICE_NAME_PATTERN.fullmatch(voice_name):
            raise InvalidVoiceNameError(
                "voice_name may contain only letters, numbers, underscores, and hyphens"
            )

        audio_path = self.voices_dir / f"{voice_name}.wav"
        transcript_path = self.voices_dir / f"{voice_name}.txt"
        missing = [str(path) for path in (audio_path, transcript_path) if not path.is_file()]
        if missing:
            raise VoiceProfileNotFoundError(
                f"Voice profile '{voice_name}' is incomplete; missing: {', '.join(missing)}"
            )

        try:
            sf.info(audio_path)
        except Exception as exc:
            raise InvalidVoiceProfileError(
                f"Reference audio for voice '{voice_name}' is not a readable WAV file"
            ) from exc

        try:
            transcript = transcript_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise InvalidVoiceProfileError(
                f"Unable to read transcript for voice '{voice_name}'"
            ) from exc
        if not transcript:
            raise InvalidVoiceProfileError(
                f"Transcript for voice '{voice_name}' must not be empty"
            )

        return audio_path, transcript

    def _infer_chunks(
        self,
        chunks: list[str],
        reference_audio: Path,
        reference_text: str,
        speed: float,
        nfe_step: int,
    ) -> tuple[np.ndarray, int]:
        if self._model is None:
            raise SynthesisError("F5-TTS model is not loaded")

        waveforms: list[np.ndarray] = []
        sample_rate: int | None = None

        for index, chunk in enumerate(chunks, start=1):
            LOGGER.debug("Rendering chunk %d/%d: %s", index, len(chunks), chunk)
            waveform, chunk_sample_rate, _ = self._model.infer(
                ref_file=str(reference_audio),
                ref_text=reference_text,
                gen_text=chunk,
                speed=speed,
                nfe_step=nfe_step,
                cross_fade_duration=0.0,
                show_info=LOGGER.debug,
            )
            chunk_audio = np.asarray(waveform, dtype=np.float32)
            if chunk_audio.ndim not in (1, 2) or chunk_audio.shape[0] == 0:
                raise SynthesisError(f"F5-TTS returned empty audio for chunk {index}")

            if sample_rate is None:
                sample_rate = int(chunk_sample_rate)
            elif sample_rate != int(chunk_sample_rate):
                raise SynthesisError("F5-TTS returned inconsistent sample rates")

            waveforms.append(chunk_audio)
            if index < len(chunks):
                silence_length = round(sample_rate * 0.25)
                silence_shape = (silence_length, *chunk_audio.shape[1:])
                waveforms.append(np.zeros(silence_shape, dtype=np.float32))

        if sample_rate is None or not waveforms:
            raise SynthesisError("F5-TTS did not generate audio")
        return np.concatenate(waveforms, axis=0), sample_rate

    @staticmethod
    def _is_out_of_memory(exc: BaseException) -> bool:
        cuda_oom = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
        return isinstance(exc, cuda_oom) or "out of memory" in str(exc).lower()

    def _release_device_cache(self) -> None:
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
