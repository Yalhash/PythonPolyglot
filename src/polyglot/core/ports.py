"""Ports: the interfaces the core needs from the outside world.

Each has at least one real adapter (``polyglot.adapters``/``polyglot.platform``) and one
fake (``polyglot.adapters.fakes``). The composition root picks implementations once,
from config. No implementation carries a "mock mode" flag.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from polyglot.core.models import ReplyOption, SuggestionContext, Transcript

RawAudio = npt.NDArray[np.float32]


class AudioSink(Protocol):
    """What an AudioSource delivers to. Implemented by the pipeline."""

    def on_audio(self, block: RawAudio) -> None:
        """Called on the audio library's thread. ``block`` is float32, shape (frames,) or
        (frames, channels), at the source's ``sample_rate``. Must return quickly."""

    def on_end(self) -> None:
        """The source has no more audio (end of file). Live sources never call this."""

    def on_error(self, error: Exception) -> None:
        """The source failed (device lost). Called at most once."""


@runtime_checkable
class AudioSource(Protocol):
    @property
    def sample_rate(self) -> int: ...

    def start(self, sink: AudioSink) -> None: ...

    def stop(self) -> None: ...


class Resampler(Protocol):
    """Streaming mono resampler from one input rate to 16 kHz. Keeps state across calls."""

    def process(self, block: RawAudio) -> RawAudio: ...

    def flush(self) -> RawAudio: ...


class VoiceActivityModel(Protocol):
    """Scores 32 ms frames (512 samples at 16 kHz) for speech probability in [0, 1]."""

    def speech_prob(self, frame: RawAudio) -> float: ...

    def reset(self) -> None: ...


class Transcriber(Protocol):
    def load(self) -> None:
        """Load the model. Called once, explicitly, before listening starts."""

    def transcribe(self, audio: RawAudio, language: str | None) -> Transcript:
        """Transcribe one utterance (16 kHz mono). Raises TranscriptionFailed."""


class Translator(Protocol):
    def translate(self, text: str, source: str | None, target: str, *, timeout: float) -> str:
        """Raises ProviderError subclasses on failure; never returns placeholder text."""


class Suggester(Protocol):
    def suggest(self, context: SuggestionContext, *, timeout: float) -> tuple[ReplyOption, ...]:
        """Raises ProviderError subclasses on failure; never returns placeholder text."""


class ShieldStatus(str, Enum):
    HIDDEN = "hidden"  # excluded from screen capture, still visible on the monitor
    UNSUPPORTED = "unsupported"  # this OS/version cannot do it (macOS 15+, Windows < 2004)
    FAILED = "failed"  # the OS call failed


class WindowShield(Protocol):
    """Hides a top-level window from screen capture/sharing while keeping it visible."""

    def exclude_from_capture(self, window_id: int) -> ShieldStatus: ...


class SecretStore(Protocol):
    def get(self, name: str) -> str | None: ...

    def set(self, name: str, value: str | None) -> None: ...
