from __future__ import annotations

import numpy as np
import pytest

from polyglot.core.models import FRAME_SAMPLES, FRAME_SECONDS, IdSource
from polyglot.core.segmenter import (
    Discarded,
    SegmentReady,
    Segmenter,
    SegmenterConfig,
    SpeechStarted,
)

FRAME = np.zeros(FRAME_SAMPLES, dtype=np.float32)


def run(seg: Segmenter, probs: list[float], t0: float = 0.0) -> list[object]:
    out: list[object] = []
    for i, p in enumerate(probs):
        frame = np.full(FRAME_SAMPLES, i, dtype=np.float32)  # frame index encoded in samples
        out += seg.push(frame, p, t0 + i * FRAME_SECONDS)
    return out


def finals(outputs: list[object]) -> list[SegmentReady]:
    return [o for o in outputs if isinstance(o, SegmentReady) and o.segment.is_final]


def frames_of(ready: SegmentReady) -> list[int]:
    audio = ready.segment.audio.reshape(-1, FRAME_SAMPLES)
    return [int(f[0]) for f in audio]


def make(**kw: object) -> Segmenter:
    return Segmenter(SegmenterConfig(**kw), IdSource())  # type: ignore[arg-type]


def test_start_needs_consecutive_frames() -> None:
    seg = make(start_frames=3)
    out = run(seg, [0.9, 0.9, 0.1, 0.9, 0.9])
    assert not any(isinstance(o, SpeechStarted) for o in out)
    out = seg.push(FRAME, 0.9, 1.0)
    assert out == [SpeechStarted(1, pytest.approx(out[0].t_start))]  # type: ignore[union-attr]
    assert seg.in_speech


def test_pre_roll_is_included_and_t_start_is_its_first_frame() -> None:
    seg = make(pre_roll_ms=96, start_frames=3, end_silence_ms=96, min_utterance_ms=0, trailing_silence_keep_ms=0)
    probs = [0.0] * 10 + [0.9] * 5 + [0.0] * 3
    out = run(seg, probs)
    started = [o for o in out if isinstance(o, SpeechStarted)]
    assert len(started) == 1
    (ready,) = finals(out)
    # pre-roll = 3 frames (96 ms) + the 3 triggering frames -> utterance starts at frame 7
    assert frames_of(ready)[0] == 7
    assert started[0].t_start == pytest.approx(7 * FRAME_SECONDS)
    assert ready.segment.t_start == pytest.approx(7 * FRAME_SECONDS)


def test_ends_after_silence_and_trims_trailing_silence() -> None:
    seg = make(pre_roll_ms=0, start_frames=1, end_silence_ms=320, trailing_silence_keep_ms=64, min_utterance_ms=0)
    probs = [0.9] * 20 + [0.0] * 10  # 10 frames = 320 ms of silence ends it
    out = run(seg, probs)
    (ready,) = finals(out)
    kept = frames_of(ready)
    assert kept == list(range(0, 22))  # 20 speech frames + 2 kept silence frames (64 ms)
    assert not seg.in_speech
    assert ready.segment.duration == pytest.approx(22 * FRAME_SECONDS)


def test_hysteresis_mid_probability_does_not_end_speech() -> None:
    seg = make(start_frames=1, end_silence_ms=160, start_threshold=0.5, end_threshold=0.35)
    out = run(seg, [0.9] * 5 + [0.4] * 30)  # below start, above end threshold
    assert finals(out) == []
    assert seg.in_speech


def test_short_noise_is_discarded() -> None:
    seg = make(start_frames=3, pre_roll_ms=300, min_utterance_ms=300, end_silence_ms=200)
    out = run(seg, [0.0] * 5 + [0.9] * 4 + [0.0] * 8)  # 4 speech frames = 128 ms
    assert [o for o in out if isinstance(o, Discarded)] == [Discarded(1)]
    assert finals(out) == []


def test_force_cut_at_quietest_frame_and_continue_as_new_utterance() -> None:
    seg = make(start_frames=1, pre_roll_ms=0, max_utterance_s=3.2, force_cut_window_s=0.64, min_utterance_ms=0)
    n_max = 100  # 3.2 s / 32 ms
    probs = [0.9] * n_max
    probs[95] = 0.36  # quietest frame inside the last 20 frames (0.64 s)
    out = run(seg, probs)
    (ready,) = finals(out)
    assert ready.segment.utterance_id == 1
    assert frames_of(ready) == list(range(0, 96))  # cut after the quietest frame
    started = [o for o in out if isinstance(o, SpeechStarted)]
    assert [s.utterance_id for s in started] == [1, 2]
    assert started[1].t_start == pytest.approx(96 * FRAME_SECONDS)
    assert seg.in_speech  # still talking: utterance 2 is open
    out = seg.flush()
    (rest,) = finals(out)
    assert rest.segment.utterance_id == 2
    assert frames_of(rest) == list(range(96, 100))


def test_partials_have_increasing_versions_then_final() -> None:
    seg = make(start_frames=1, pre_roll_ms=0, partial_interval_ms=320, end_silence_ms=320, min_utterance_ms=0)
    out = run(seg, [0.9] * 35 + [0.0] * 10)
    partials = [o.segment for o in out if isinstance(o, SegmentReady) and not o.segment.is_final]
    assert [p.version for p in partials] == [0, 1, 2]
    assert all(p.utterance_id == 1 for p in partials)
    assert [len(p.audio) for p in partials] == sorted(len(p.audio) for p in partials)
    assert len(finals(out)) == 1


def test_flush_finishes_open_utterance_and_is_idempotent() -> None:
    seg = make(start_frames=1, pre_roll_ms=0, min_utterance_ms=0)
    run(seg, [0.9] * 10)
    out = seg.flush()
    assert len(finals(out)) == 1
    assert seg.flush() == []


def test_ids_increase_across_utterances() -> None:
    seg = make(start_frames=1, pre_roll_ms=0, end_silence_ms=96, min_utterance_ms=0)
    out = run(seg, ([0.9] * 5 + [0.0] * 4) * 3)
    assert [f.segment.utterance_id for f in finals(out)] == [1, 2, 3]


@pytest.mark.parametrize(
    "kw",
    [
        {"start_threshold": 0.3, "end_threshold": 0.5},
        {"start_frames": 0},
        {"max_utterance_s": 1.0, "force_cut_window_s": 2.0},
    ],
)
def test_invalid_config_rejected(kw: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SegmenterConfig(**kw)  # type: ignore[arg-type]


def test_rejects_wrong_frame_size() -> None:
    with pytest.raises(ValueError):
        make().push(np.zeros(100, dtype=np.float32), 0.9, 0.0)
