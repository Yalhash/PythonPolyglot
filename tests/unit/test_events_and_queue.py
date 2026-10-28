from __future__ import annotations

import threading

import numpy as np

from polyglot.core.events import EventBus, PipelineState, PipelineStatus
from polyglot.core.models import SpeechSegment
from polyglot.core.pipeline import AsrQueue


def seg(uid: int, final: bool = True, version: int = 0) -> SpeechSegment:
    return SpeechSegment(uid, np.zeros(4, dtype=np.float32), 0.0, 1.0, final, version)


# ------------------------------------------------------------------ EventBus
def test_bus_delivers_to_all_and_unsubscribes() -> None:
    bus = EventBus()
    a: list[object] = []
    b: list[object] = []
    unsub_a = bus.subscribe(a.append)
    bus.subscribe(b.append)
    ev = PipelineStatus(PipelineState.LISTENING)
    bus.publish(ev)
    unsub_a()
    bus.publish(ev)
    assert a == [ev]
    assert b == [ev, ev]


def test_failing_subscriber_does_not_break_others() -> None:
    bus = EventBus()
    got: list[object] = []

    def boom(_: object) -> None:
        raise RuntimeError("subscriber bug")

    bus.subscribe(boom)
    bus.subscribe(got.append)
    bus.publish(PipelineStatus(PipelineState.STOPPED))
    assert len(got) == 1


def test_bus_is_safe_under_concurrent_subscribe_and_publish() -> None:
    bus = EventBus()
    counts: list[int] = []
    stop = threading.Event()

    def publisher() -> None:
        while not stop.is_set():
            bus.publish(PipelineStatus(PipelineState.LISTENING))

    t = threading.Thread(target=publisher)
    t.start()
    for _ in range(200):
        unsub = bus.subscribe(lambda _e: counts.append(1))
        unsub()
    stop.set()
    t.join()


# ------------------------------------------------------------------ AsrQueue
def test_finals_fifo_and_bounded_drop_oldest() -> None:
    q = AsrQueue(max_finals=2)
    assert q.put_final(seg(1)) is None
    assert q.put_final(seg(2)) is None
    dropped = q.put_final(seg(3))
    assert dropped is not None and dropped.utterance_id == 1
    assert [q.get(0).utterance_id, q.get(0).utterance_id] == [2, 3]  # type: ignore[union-attr]
    assert q.get(0) is None


def test_partial_slot_keeps_only_latest_per_utterance() -> None:
    q = AsrQueue(4)
    q.put_partial(seg(5, final=False, version=0))
    q.put_partial(seg(5, final=False, version=1))
    q.put_partial(seg(6, final=False, version=0))
    assert q.depth() == 2
    first = q.get(0)
    assert first is not None and (first.utterance_id, first.version) == (5, 1)


def test_finals_are_served_before_partials_and_replace_them() -> None:
    q = AsrQueue(4)
    q.put_partial(seg(7, final=False))
    q.put_final(seg(6))
    q.put_final(seg(7))  # the final for 7 makes its partial pointless
    got = [q.get(0) for _ in range(3)]
    assert [(g.utterance_id, g.is_final) for g in got if g] == [(6, True), (7, True)]


def test_close_wakes_waiters() -> None:
    q = AsrQueue(1)
    threading.Timer(0.02, q.close).start()
    assert q.get(2.0) is None
    assert q.closed
