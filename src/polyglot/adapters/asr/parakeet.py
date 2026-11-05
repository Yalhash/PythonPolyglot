"""Local speech recognition with NVIDIA Parakeet TDT v3, run through ONNX Runtime.

``onnx-asr`` runs the exported model on ONNX Runtime, so no NVIDIA hardware or NeMo/PyTorch
is needed: it runs on any CPU (int8 by default), on CUDA when ``onnxruntime-gpu`` is
installed, and on any DirectX 12 GPU (AMD, Intel) when ``onnxruntime-directml`` is.

Parakeet v3 covers 25 European languages and picks the language itself. It does not
report which one it heard, so the transcript carries the language the pipeline asked
for (the locked or configured one), or none.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from polyglot.core.errors import ConfigError, TranscriptionFailed
from polyglot.core.models import SAMPLE_RATE, Transcript
from polyglot.core.ports import RawAudio

log = logging.getLogger(__name__)

DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v3"

#: ISO 639-1 codes of the 25 languages Parakeet TDT 0.6B v3 transcribes.
LANGUAGES = frozenset(
    [
        "bg",
        "hr",
        "cs",
        "da",
        "nl",
        "en",
        "et",
        "fi",
        "fr",
        "de",
        "el",
        "hu",
        "it",
        "lv",
        "lt",
        "mt",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "es",
        "sv",
        "ru",
        "uk",
    ]
)

_CPU, _CUDA, _DML = "CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"


def choose_providers(device: str, available: list[str]) -> list[str]:
    """ONNX Runtime providers for a ``device`` setting, fastest first, CPU always last.

    ``auto`` uses a GPU provider when the installed onnxruntime build has one (CUDA, then
    DirectML). ``cuda`` insists on CUDA, so a missing build is an error, not a slow
    surprise."""
    if device == "cpu":
        return [_CPU]
    if device == "cuda":
        if _CUDA not in available:
            raise ConfigError(
                "Parakeet on CUDA needs the onnxruntime-gpu package; "
                "choose Automatic or CPU instead"
            )
        return [_CUDA, _CPU]
    gpu = [p for p in (_CUDA, _DML) if p in available]
    return [*gpu[:1], _CPU]


class ParakeetTranscriber:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        quantization: str | None = "int8",
        download_root: str | None = None,
    ) -> None:
        self.model_name, self.device = model, device
        self.quantization = quantization or None
        self.download_root = download_root
        self._model: Any = None
        self.effective_device: str | None = None  # the provider ONNX Runtime uses, after load()

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import onnx_asr  # lazy: optional dependency (uv sync --extra parakeet)
            import onnxruntime
        except ImportError as exc:
            raise ConfigError(
                "Parakeet needs the onnx-asr package (uv sync --extra parakeet)"
            ) from exc
        providers = choose_providers(self.device, list(onnxruntime.get_available_providers()))
        log.info(
            "loading Parakeet %s (%s) with %s",
            self.model_name,
            self.quantization or "full precision",
            providers[0],
        )
        try:
            self._model = onnx_asr.load_model(
                self.model_name,
                self.download_root,
                quantization=self.quantization,
                providers=providers,
            )
        except Exception as exc:
            raise TranscriptionFailed(
                f"could not load Parakeet model {self.model_name!r}: {exc}"
            ) from exc
        self.effective_device = providers[0].removesuffix("ExecutionProvider").lower()
        log.info("Parakeet %s loaded on %s", self.model_name, self.effective_device)

    def transcribe(self, audio: RawAudio, language: str | None) -> Transcript:
        if self._model is None:
            raise TranscriptionFailed("model not loaded")
        try:
            result = self._model.recognize(
                np.ascontiguousarray(audio, dtype=np.float32), sample_rate=SAMPLE_RATE
            )
        except Exception as exc:
            raise TranscriptionFailed(f"transcription failed: {exc}") from exc
        text = str(getattr(result, "text", result) or "").strip()
        # No language detection to report: echo what the pipeline already decided.
        return Transcript(
            text=text,
            language=language,
            language_probability=1.0 if language else 0.0,
        )
