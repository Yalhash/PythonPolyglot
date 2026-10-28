"""Local speech recognition with faster-whisper (CTranslate2)."""

from __future__ import annotations

import logging
from typing import Any

from polyglot.core.errors import ConfigError, TranscriptionFailed
from polyglot.core.models import Transcript
from polyglot.core.ports import RawAudio

log = logging.getLogger(__name__)

MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")


class FasterWhisperTranscriber:
    def __init__(
        self,
        model: str = "base",
        *,
        device: str = "auto",
        compute_type: str = "default",
        beam_size: int = 5,
        download_root: str | None = None,
    ) -> None:
        self.model_name, self.device, self.compute_type = model, device, compute_type
        self.beam_size, self.download_root = beam_size, download_root
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel  # lazy: heavy native dependency
        except ImportError as exc:
            raise ConfigError("faster-whisper is not installed") from exc
        log.info("loading Whisper %s on %s (%s)", self.model_name, self.device, self.compute_type)
        try:
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )
        except Exception as exc:
            raise TranscriptionFailed(
                f"could not load Whisper model {self.model_name!r}: {exc}"
            ) from exc

    def transcribe(self, audio: RawAudio, language: str | None) -> Transcript:
        if self._model is None:
            raise TranscriptionFailed("model not loaded")
        try:
            segments, info = self._model.transcribe(
                audio,
                language=language,
                beam_size=self.beam_size,
                vad_filter=False,  # the pipeline already segmented on speech
                condition_on_previous_text=False,  # each utterance stands alone
                without_timestamps=True,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as exc:
            raise TranscriptionFailed(f"transcription failed: {exc}") from exc
        return Transcript(
            text=text,
            language=getattr(info, "language", None),
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        )
