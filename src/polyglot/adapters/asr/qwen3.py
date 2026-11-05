"""Local speech recognition with Qwen3-ASR (0.6B or 1.7B), via the ``qwen-asr`` package.

Runs on PyTorch: on a CUDA GPU when one is available, otherwise on the CPU in float32
(no NVIDIA hardware needed; the 0.6B model is the practical size on a CPU). Qwen3-ASR
covers 30 languages plus Chinese dialects, and reports the language it heard.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from polyglot.core import languages
from polyglot.core.errors import ConfigError, TranscriptionFailed
from polyglot.core.models import SAMPLE_RATE, Transcript
from polyglot.core.ports import RawAudio

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
MODELS = ("Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B")

#: Language code -> the name Qwen3-ASR takes and reports. Its 30 languages.
LANGUAGES: dict[str, str] = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German",
    "fr": "French", "es": "Spanish", "pt": "Portuguese", "id": "Indonesian",
    "it": "Italian", "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
    "ja": "Japanese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch",
    "sv": "Swedish", "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "tl": "Filipino", "fa": "Persian", "el": "Greek", "hu": "Hungarian",
    "mk": "Macedonian", "ro": "Romanian",
}  # fmt: skip
_CODES = {name.lower(): code for code, name in LANGUAGES.items()}
_CODES["fil"] = "tl"  # Filipino's ISO 639-2 code, should the model report it


def language_name(code: str | None) -> str | None:
    """The name to force ``code`` with, or None (auto-detect) if Qwen3-ASR lacks it."""
    if not code:
        return None
    code = code.replace("_", "-").lower()
    if code in ("fil", "zh-yue", "zh-hk"):
        code = "tl" if code == "fil" else "yue"
    return LANGUAGES.get(code) or LANGUAGES.get(languages.primary(code) or "")


def language_code(reported: str | None) -> str | None:
    """Qwen3-ASR's reported language ("French", sometimes "fr") as a code, or None."""
    if not reported:
        return None
    key = reported.strip().lower()
    if key in LANGUAGES:
        return key
    return _CODES.get(key)


class Qwen3Transcriber:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        self.model_name, self.device, self.max_new_tokens = model, device, max_new_tokens
        self._model: Any = None
        self.effective_device: str | None = None  # "cpu" or "cuda:0", after load()

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # lazy: optional dependency (uv sync --extra qwen3)
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ConfigError(
                "Qwen3-ASR needs the qwen-asr package (uv sync --extra qwen3)"
            ) from exc
        has_cuda = bool(torch.cuda.is_available())
        if self.device == "cuda" and not has_cuda:
            raise ConfigError("no CUDA GPU is available to PyTorch; choose Automatic or CPU")
        on_gpu = has_cuda and self.device != "cpu"
        device_map = "cuda:0" if on_gpu else "cpu"
        # bfloat16 halves GPU memory; on a CPU it is slow or unsupported, so use float32.
        dtype = torch.bfloat16 if on_gpu else torch.float32
        log.info("loading Qwen3-ASR %s on %s", self.model_name, device_map)
        try:
            self._model = Qwen3ASRModel.from_pretrained(
                self.model_name,
                dtype=dtype,
                device_map=device_map,
                max_inference_batch_size=1,  # one utterance at a time
                max_new_tokens=self.max_new_tokens,
            )
        except Exception as exc:
            raise TranscriptionFailed(
                f"could not load Qwen3-ASR model {self.model_name!r}: {exc}"
            ) from exc
        self.effective_device = device_map
        log.info("Qwen3-ASR %s loaded on %s", self.model_name, device_map)

    def transcribe(self, audio: RawAudio, language: str | None) -> Transcript:
        if self._model is None:
            raise TranscriptionFailed("model not loaded")
        forced = language_name(language)
        try:
            results = self._model.transcribe(
                audio=(np.ascontiguousarray(audio, dtype=np.float32), SAMPLE_RATE),
                language=forced,
            )
            result = results[0] if isinstance(results, (list, tuple)) else results
        except Exception as exc:
            raise TranscriptionFailed(f"transcription failed: {exc}") from exc
        text = str(getattr(result, "text", "") or "").strip()
        code = language_code(getattr(result, "language", None)) if text else None
        if forced and code is None:
            code = language  # forced: the model did not need to say
        # Qwen3-ASR gives no confidence score. A reported language counts as confident;
        # the lock still waits for ``asr.lock_after`` utterances that agree.
        return Transcript(text=text, language=code, language_probability=1.0 if code else 0.0)
