"""Domain model: the concepts Polyglot works with, independent of any library or UI.

Everything here is immutable. A change produces a new object (``dataclasses.replace``),
so a value handed to another thread can never change underneath it.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np
import numpy.typing as npt

from polyglot.core.profile import UserProfile

#: Every stage after the front end works at this rate, mono, float32 in [-1, 1].
SAMPLE_RATE = 16_000
#: One VAD frame: 512 samples at 16 kHz = 32 ms (the frame size Silero VAD expects).
FRAME_SAMPLES = 512
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE

AudioArray = npt.NDArray[np.float32]


class UtteranceStatus(str, Enum):
    """Lifecycle of one utterance. See the design doc, "Domain model"."""

    LISTENING = "listening"  # speech started, no text yet
    PARTIAL = "partial"  # provisional text, may still change
    FINAL = "final"  # endpoint reached, text fixed
    TRANSLATED = "translated"  # translation attached
    FAILED = "failed"  # a stage failed; ``error`` says which way


class Speaker(str, Enum):
    """Who spoke an utterance. Replies are suggested for ``OTHER``; ``USER`` turns (from
    the user's own microphone, once there is one) only add context."""

    OTHER = "other"
    USER = "user"


class ErrorKind(str, Enum):
    """Why something failed. Shown to the user; never replaced with made-up content."""

    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    QUOTA = "quota"
    AUTH = "auth"
    BAD_RESPONSE = "bad_response"
    TRANSCRIPTION = "transcription"
    OVERLOADED = "overloaded"
    NOT_CONFIGURED = "not_configured"
    NOTHING_TO_ANSWER = "nothing_to_answer"
    AUDIO = "audio"
    INTERNAL = "internal"


@dataclass(frozen=True, eq=False)
class SpeechSegment:
    """Audio for one utterance, produced by the segmenter.

    ``is_final`` segments are complete utterances. Non-final segments are partials: the
    utterance so far, re-sent as it grows, each with a higher ``version``.
    """

    utterance_id: int
    audio: AudioArray
    t_start: float
    t_end: float
    is_final: bool
    version: int = 0
    speaker: Speaker = Speaker.OTHER  # which audio stream it came from

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None
    language_probability: float = 0.0


@dataclass(frozen=True)
class Utterance:
    """One continuous stretch of speech by one speaker, bounded by pauses."""

    id: int
    t_start: float
    t_end: float | None = None
    status: UtteranceStatus = UtteranceStatus.LISTENING
    text: str = ""
    source_lang: str | None = None
    translation: str | None = None
    error: ErrorKind | None = None
    partial_version: int = -1
    speaker: Speaker = Speaker.OTHER

    def evolve(self, **changes: object) -> Utterance:
        return replace(self, **changes)  # type: ignore[arg-type]

    @property
    def is_final(self) -> bool:
        return self.status in (UtteranceStatus.FINAL, UtteranceStatus.TRANSLATED) or (
            self.status is UtteranceStatus.FAILED and self.t_end is not None
        )


@dataclass(frozen=True)
class ReplyOption:
    """One suggested reply: what to say (their language) and what it means (yours)."""

    source_text: str
    user_text: str


@dataclass(frozen=True)
class Suggestion:
    """Replies to one utterance, kept linked to it."""

    utterance_id: int
    options: tuple[ReplyOption, ...] = ()
    error: ErrorKind | None = None
    message: str = ""


class ConversationType(str, Enum):
    INTERVIEW = "interview"
    MEETING = "meeting"
    CASUAL = "casual"


@dataclass(frozen=True)
class SuggestionContext:
    """Everything a Suggester needs; built by the coordinator from the session."""

    target: Utterance
    history: tuple[Utterance, ...]
    source_language: str  # full name, e.g. "Spanish"
    user_language: str  # full name, e.g. "English"
    conversation_type: ConversationType = ConversationType.MEETING
    max_options: int = 2
    profile: UserProfile = field(default_factory=UserProfile)


class IdSource:
    """Monotonic utterance IDs, starting at 1. IDs define spoken order.

    One source serves every audio stream (the other person's and the user's microphone),
    each on its own thread, so an ID is drawn under a lock."""

    def __init__(self, start: int = 1) -> None:
        self._next = itertools.count(start)
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return next(self._next)
