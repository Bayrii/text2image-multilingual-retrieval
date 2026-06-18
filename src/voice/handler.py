"""Whisper-backed transcription used by both the FastAPI route and the Gradio app.

Accepts raw numpy audio (from gradio.Audio) or compressed bytes (from a file
upload), normalizes to 16 kHz mono float32, and runs Whisper.

The same instance can be shared across the two interfaces — Whisper is loaded
once at process startup and is thread-safe under torch.no_grad.
"""
from __future__ import annotations

import io
from typing import Any, Dict, Optional

import numpy as np

from ..utils import get_logger

logger = get_logger("voice")

WHISPER_SR = 16000


class VoiceHandler:
    def __init__(self, model_size: str = "small", device: Optional[str] = None):
        import torch
        import whisper
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Whisper '%s' on %s ...", model_size, device)
        self.model = whisper.load_model(model_size, device=device)
        self.device = device
        self.model_size = model_size

    @staticmethod
    def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
            peak = np.max(np.abs(audio)) if audio.size else 0.0
            if peak > 1.5:
                audio = audio / peak
        return audio

    @staticmethod
    def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == WHISPER_SR:
            return audio
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), WHISPER_SR)
        return resample_poly(audio, WHISPER_SR // g, sr // g).astype(np.float32)

    def transcribe(self, audio: np.ndarray, sr: int,
                   language: Optional[str] = None) -> Dict[str, Any]:
        audio = self._to_mono_float32(audio)
        audio = self._resample(audio, sr)
        if audio.size < WHISPER_SR // 4:
            return {"text": "", "language": language or "unknown",
                    "warning": "Audio too short (< 0.25 s)"}
        result = self.model.transcribe(audio, language=language,
                                       fp16=(self.device == "cuda"))
        return {
            "text": (result.get("text") or "").strip(),
            "language": result.get("language", language or "unknown"),
        }

    def transcribe_bytes(self, audio_bytes: bytes,
                         language: Optional[str] = None) -> Dict[str, Any]:
        import soundfile as sf
        try:
            arr, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception as e:
            raise ValueError(f"Could not decode audio (need WAV/FLAC/OGG): {e}") from e
        return self.transcribe(arr, sr, language=language)
