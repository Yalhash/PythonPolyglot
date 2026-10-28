"""Events the pipeline publishes, and the bus that carries them.

Subscribers run synchronously on the publishing thread. A subscriber that touches a UI
toolkit must hand the event over to its own thread (the Qt presenter does this with a
queued signal). A failing subscriber is logged and never breaks the publisher.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from polyglot.core.models import ErrorKind, Suggestion, Utterance

log = logging.getLogger(__name__)


class PipelineState(str, Enum):
    STOPPED = "stopped"
    LOADING = "loading"
    LISTENING = "listening"
    FALLING_BEHIND = "falling_behind"
    ERROR = "error"


@dataclass(frozen=True)
class UtteranceUpdated:
    """An utterance was created or changed (started, partial, final, translated, failed)."""

    utterance: Utterance


@dataclass(frozen=True)
class UtteranceDiscarded:
    """An utterance turned out not to be speech (too short, or no words). Remove its row."""

    utterance_id: int


@dataclass(frozen=True)
class SuggestionStarted:
    utterance_id: int


@dataclass(frozen=True)
class SuggestionsReady:
    suggestion: Suggestion


@dataclass(frozen=True)
class LanguageChanged:
    """Source language was locked, unlocked or overridden."""

    language: str | None
    locked: bool


@dataclass(frozen=True)
class PipelineStatus:
    state: PipelineState
    detail: str = ""


@dataclass(frozen=True)
class StageError:
    stage: str
    kind: ErrorKind
    message: str
    retryable: bool = False


Event = (
    UtteranceUpdated
    | UtteranceDiscarded
    | SuggestionStarted
    | SuggestionsReady
    | LanguageChanged
    | PipelineStatus
    | StageError
)
Subscriber = Callable[[Event], None]


class EventBus:
    """In-process publish/subscribe. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: tuple[Subscriber, ...] = ()

    def subscribe(self, subscriber: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subscribers = (*self._subscribers, subscriber)

        def unsubscribe() -> None:
            with self._lock:
                self._subscribers = tuple(s for s in self._subscribers if s is not subscriber)

        return unsubscribe

    def publish(self, event: Event) -> None:
        for subscriber in self._subscribers:  # tuple snapshot: safe without the lock
            try:
                subscriber(event)
            except Exception:
                log.exception("event subscriber failed on %s", type(event).__name__)
