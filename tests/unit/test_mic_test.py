from __future__ import annotations

import threading

import numpy as np

from polyglot.core.errors import AudioSourceError
from polyglot.core.models import AudioClip
from polyglot.ui.mic_test import MicTestRunner, MicTestUpdate
from tests.conftest import wait_for


def clip(peak: float, seconds: float = 0.1, rate: int = 16_000) -> AudioClip:
    samples = np.zeros(int(seconds * rate), dtype=np.float32)
    samples[0] = peak
    return AudioClip(samples, rate)


class FakeTester:
    def __init__(self, recorded: AudioClip, error: Exception | None = None) -> None:
        self.clip, self.error = recorded, error
        self.recorded: list[tuple[object, float]] = []
        self.played: list[AudioClip] = []
        self.record_thread: str | None = None
        self.gate = threading.Event()
        self.gate.set()

    def record(self, device: int | str | None, seconds: float) -> AudioClip:
        self.record_thread = threading.current_thread().name
        self.recorded.append((device, seconds))
        self.gate.wait(5)
        if self.error is not None:
            raise self.error
        return self.clip

    def play(self, played: AudioClip) -> None:
        self.played.append(played)


def run(tester: FakeTester, device: object = None) -> tuple[MicTestRunner, list[MicTestUpdate]]:
    updates: list[MicTestUpdate] = []
    runner = MicTestRunner(tester, updates.append, seconds=5.0)
    assert runner.start(device)  # type: ignore[arg-type]
    assert wait_for(lambda: not runner.running)
    return runner, updates


def test_records_then_plays_back_the_same_clip() -> None:
    tester = FakeTester(clip(0.4))
    _, updates = run(tester, device="3")
    assert tester.recorded == [(3, 5.0)]  # a numeric id becomes a number; 5 seconds
    assert tester.played == [tester.clip]
    assert [(u.busy, u.level) for u in updates] == [(True, "info"), (True, "info"), (False, "info")]
    assert "Recording" in updates[0].text and "Playing" in updates[1].text
    assert updates[-1].text == "Sounds good."
    assert tester.record_thread == "polyglot-mic-test"  # never blocks the caller


def test_the_default_device_is_passed_through_as_none() -> None:
    tester = FakeTester(clip(0.4))
    run(tester, device=None)
    assert tester.recorded[0][0] is None
    run(tester, device="Headset Mic")
    assert tester.recorded[1][0] == "Headset Mic"


def test_silence_is_reported_and_not_played() -> None:
    tester = FakeTester(clip(0.0))
    _, updates = run(tester)
    assert tester.played == []
    assert updates[-1].level == "warn" and "No sound" in updates[-1].text and not updates[-1].busy


def test_quiet_and_clipping_recordings_are_played_but_flagged() -> None:
    for peak, word in ((0.03, "quiet"), (1.0, "distorting")):
        tester = FakeTester(clip(peak))
        _, updates = run(tester)
        assert tester.played and updates[-1].level == "warn" and word in updates[-1].text


def test_a_failing_device_is_an_error_and_the_test_can_run_again() -> None:
    tester = FakeTester(clip(0.4), error=AudioSourceError("device busy"))
    runner, updates = run(tester)
    assert updates[-1].level == "error" and "device busy" in updates[-1].text
    assert not updates[-1].busy and tester.played == []
    tester.error = None
    assert runner.start(None)  # not stuck "running"
    assert wait_for(lambda: not runner.running)
    assert tester.played


def test_an_unexpected_driver_error_still_frees_the_button() -> None:
    _, updates = run(FakeTester(clip(0.4), error=OSError("portaudio blew up")))
    assert updates[-1].level == "error" and not updates[-1].busy


def test_only_one_test_runs_at_a_time() -> None:
    tester = FakeTester(clip(0.4))
    tester.gate.clear()
    updates: list[MicTestUpdate] = []
    runner = MicTestRunner(tester, updates.append)
    assert runner.start(None)
    assert not runner.start(None)  # still recording
    tester.gate.set()
    assert wait_for(lambda: not runner.running)
    assert len(tester.recorded) == 1


def test_cancel_skips_playback_and_stays_quiet() -> None:
    tester = FakeTester(clip(0.4))
    tester.gate.clear()
    updates: list[MicTestUpdate] = []
    runner = MicTestRunner(tester, updates.append)
    runner.start(None)
    assert wait_for(lambda: len(updates) == 1)  # "Recording…"
    runner.cancel()  # the dialog was closed
    tester.gate.set()
    assert wait_for(lambda: not runner.running)
    assert tester.played == [] and len(updates) == 1


def test_clip_level_thresholds() -> None:
    assert [clip(p).level() for p in (0.0, 0.005, 0.02, 0.3, 0.99)] == [
        "silent",
        "silent",
        "quiet",
        "ok",
        "clipping",
    ]
    c = clip(0.5, seconds=2.0, rate=8_000)
    assert c.duration == 2.0 and c.peak == 0.5
    assert AudioClip(np.zeros(0, dtype=np.float32), 16_000).level() == "silent"
