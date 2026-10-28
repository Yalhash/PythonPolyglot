"""Working fake implementations of every port, chosen by wiring rather than flags.

Used by the tests and by ``polyglot --demo`` (run the whole app with no models, keys or
network).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from polyglot.core.errors import PolyglotError
from polyglot.core.models import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    ReplyOption,
    SuggestionContext,
    Transcript,
)
from polyglot.core.ports import RawAudio, ShieldStatus


class ScriptedVad:
    """Returns a fixed sequence of probabilities, then ``default``."""

    def __init__(self, probs: Iterable[float], default: float = 0.0) -> None:
        self._script = list(probs)
        self._i = 0
        self.default = default

    def speech_prob(self, frame: RawAudio) -> float:
        if self._i < len(self._script):
            p = self._script[self._i]
            self._i += 1
            return p
        return self.default

    def reset(self) -> None:
        """A script is a one-off: reset does not rewind it."""


class AmplitudeVad:
    """Speech iff the frame's peak exceeds ``threshold``. Deterministic for synthetic audio."""

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    def speech_prob(self, frame: RawAudio) -> float:
        return 1.0 if frame.size and float(np.max(np.abs(frame))) > self.threshold else 0.0

    def reset(self) -> None:
        pass


class ScriptedTranscriber:
    """Returns canned transcripts, in order; optional delay or failure per call."""

    def __init__(
        self,
        texts: Sequence[str] | Callable[[RawAudio], str] = (),
        *,
        language: str = "es",
        probability: float = 0.95,
        delay_s: float = 0.0,
        fail_on: Iterable[int] = (),
        error: PolyglotError | None = None,
        load_error: PolyglotError | None = None,
    ) -> None:
        self._texts = texts
        self.language, self.probability, self.delay_s = language, probability, delay_s
        self._fail_on = set(fail_on)
        self._error = error or PolyglotError("scripted failure")
        self._load_error = load_error
        self.calls: list[tuple[int, str | None]] = []  # (audio samples, language hint)
        self.loaded = False
        self._lock = threading.Lock()
        self.max_concurrency = 0
        self._active = 0

    def load(self) -> None:
        if self._load_error is not None:
            raise self._load_error
        self.loaded = True

    def transcribe(self, audio: RawAudio, language: str | None) -> Transcript:
        with self._lock:
            n = len(self.calls)
            self.calls.append((int(audio.shape[0]), language))
            self._active += 1
            self.max_concurrency = max(self.max_concurrency, self._active)
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            if n in self._fail_on:
                raise self._error
            if callable(self._texts):
                text = self._texts(audio)
            else:
                text = self._texts[n] if n < len(self._texts) else f"utterance {n + 1}"
            return Transcript(text, language or self.language, self.probability)
        finally:
            with self._lock:
                self._active -= 1


class EchoTranslator:
    """``"[es→en] hola"``; optional per-text delay or error."""

    def __init__(
        self,
        delays: dict[str, float] | None = None,
        errors: dict[str, PolyglotError] | None = None,
    ) -> None:
        self.delays = delays or {}
        self.errors = errors or {}
        self.calls: list[tuple[str, str | None, str]] = []

    def translate(self, text: str, source: str | None, target: str, *, timeout: float) -> str:
        self.calls.append((text, source, target))
        time.sleep(self.delays.get(text, 0.0))
        if text in self.errors:
            raise self.errors[text]
        return f"[{source or '?'}→{target}] {text}"


class CannedSuggester:
    def __init__(self, delay_s: float = 0.0, error: PolyglotError | None = None) -> None:
        self.delay_s, self.error = delay_s, error
        self.contexts: list[SuggestionContext] = []

    def suggest(self, context: SuggestionContext, *, timeout: float) -> tuple[ReplyOption, ...]:
        self.contexts.append(context)
        time.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        return tuple(
            ReplyOption(f"respuesta {i + 1} a: {context.target.text}", f"reply {i + 1}")
            for i in range(context.max_options)
        )


class NoopShield:
    def __init__(self, status: ShieldStatus = ShieldStatus.UNSUPPORTED) -> None:
        self.status = status
        self.calls: list[int] = []

    def exclude_from_capture(self, window_id: int) -> ShieldStatus:
        self.calls.append(window_id)
        return self.status


def tone(
    seconds: float, freq: float = 220.0, amp: float = 0.3, rate: int = SAMPLE_RATE
) -> RawAudio:
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float, rate: int = SAMPLE_RATE) -> RawAudio:
    return np.zeros(int(seconds * rate), dtype=np.float32)


def speech_like(pattern: Sequence[tuple[str, float]], rate: int = SAMPLE_RATE) -> RawAudio:
    """Build audio from ("tone"|"silence", seconds) pairs, padded to whole frames."""
    parts = [tone(s, rate=rate) if kind == "tone" else silence(s, rate=rate) for kind, s in pattern]
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    pad = (-audio.size) % FRAME_SAMPLES if rate == SAMPLE_RATE else 0
    return np.concatenate((audio, np.zeros(pad, dtype=np.float32)))
