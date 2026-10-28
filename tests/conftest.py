"""Shared test helpers. Everything here uses fakes: no models, devices or network."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from polyglot.adapters.audio.playback import ArraySource
from polyglot.adapters.audio.resample import IdentityResampler
from polyglot.adapters.fakes import AmplitudeVad, ScriptedTranscriber, speech_like
from polyglot.core.events import Event, EventBus, UtteranceUpdated
from polyglot.core.models import SAMPLE_RATE, Utterance, UtteranceStatus
from polyglot.core.pipeline import Pipeline, PipelineConfig
from polyglot.core.ports import Suggester, Translator


class Recorder:
    """Collects bus events from any thread."""

    def __init__(self, bus: EventBus) -> None:
        self._lock = threading.Lock()
        self.events: list[Event] = []
        bus.subscribe(self._on)

    def _on(self, ev: Event) -> None:
        with self._lock:
            self.events.append(ev)

    def of(self, kind: type) -> list[Any]:
        with self._lock:
            return [e for e in self.events if isinstance(e, kind)]

    def utterances(self) -> dict[int, Utterance]:
        """Latest state of each utterance."""
        out: dict[int, Utterance] = {}
        for ev in self.of(UtteranceUpdated):
            out[ev.utterance.id] = ev.utterance
        return out

    def finals(self) -> list[Utterance]:
        return [
            u
            for _, u in sorted(self.utterances().items())
            if u.status in (UtteranceStatus.FINAL, UtteranceStatus.TRANSLATED)
        ]


def wait_for(predicate: Callable[[], bool], timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def utterances_audio(n: int, speech_s: float = 1.0, gap_s: float = 0.9) -> Any:
    """n bursts of 'speech' (tone) separated by silences long enough to end each one."""
    pattern: list[tuple[str, float]] = [("silence", 0.3)]
    for _ in range(n):
        pattern += [("tone", speech_s), ("silence", gap_s)]
    return speech_like(pattern)


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def recorder(bus: EventBus) -> Recorder:
    return Recorder(bus)


@pytest.fixture
def make_pipeline(bus: EventBus) -> Iterator[Callable[..., Pipeline]]:
    made: list[Pipeline] = []

    def factory(
        audio: Any = None,
        *,
        transcriber: ScriptedTranscriber | None = None,
        translator: Translator | None = None,
        suggester: Suggester | None = None,
        config: PipelineConfig | None = None,
        source: Any = None,
        vad: Any = None,
        **kw: Any,
    ) -> Pipeline:
        if source is None:
            source = ArraySource(audio if audio is not None else utterances_audio(2), SAMPLE_RATE)
        cfg = config or PipelineConfig(ring_seconds=600.0)
        p = Pipeline(
            source=source,
            resampler_factory=lambda _rate: IdentityResampler(),
            vad=vad or AmplitudeVad(),
            transcriber=transcriber or ScriptedTranscriber(),
            translator=translator,
            suggester=suggester,
            bus=bus,
            config=cfg,
            **kw,
        )
        made.append(p)
        return p

    yield factory
    for p in made:
        p.close()
