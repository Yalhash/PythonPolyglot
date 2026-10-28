"""Endpointing: turn a stream of (frame, speech probability) into utterances.

Pure logic with no model and no threads, so every rule is unit-testable with scripted
probabilities. Rules (all configurable, see ``SegmenterConfig``):

* start: probability >= ``start_threshold`` for ``start_frames`` frames in a row; the
  utterance includes ``pre_roll_ms`` of audio from before the start.
* end: probability < ``end_threshold`` for ``end_silence_ms`` in a row (hysteresis:
  the end threshold is lower than the start threshold, so noise near one value does not
  flicker the state).
* force-cut: an utterance reaching ``max_utterance_s`` is cut at the least speech-like
  frame in its last ``force_cut_window_s``; the remainder continues as a new utterance.
* discard: utterances with less than ``min_utterance_ms`` of speech are dropped.
* partials: optionally re-emit the growing utterance every ``partial_interval_ms``.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from polyglot.core.models import FRAME_SAMPLES, FRAME_SECONDS, AudioArray, SpeechSegment


@dataclass(frozen=True)
class SegmenterConfig:
    start_threshold: float = 0.5
    end_threshold: float = 0.35
    start_frames: int = 3
    end_silence_ms: int = 600
    pre_roll_ms: int = 300
    max_utterance_s: float = 15.0
    force_cut_window_s: float = 2.0
    min_utterance_ms: int = 300
    trailing_silence_keep_ms: int = 100
    partial_interval_ms: int = 0  # 0 disables partials

    def __post_init__(self) -> None:
        if not 0.0 <= self.end_threshold <= self.start_threshold <= 1.0:
            raise ValueError("need 0 <= end_threshold <= start_threshold <= 1")
        if self.start_frames < 1:
            raise ValueError("start_frames must be >= 1")
        if self.max_utterance_s <= self.force_cut_window_s:
            raise ValueError("max_utterance_s must exceed force_cut_window_s")


def _frames(ms: float) -> int:
    return max(0, math.ceil(ms / 1000.0 / FRAME_SECONDS))


@dataclass(frozen=True)
class SpeechStarted:
    utterance_id: int
    t_start: float


@dataclass(frozen=True)
class SegmentReady:
    segment: SpeechSegment


@dataclass(frozen=True)
class Discarded:
    utterance_id: int


SegmenterOutput = SpeechStarted | SegmentReady | Discarded


class Segmenter:
    def __init__(self, config: SegmenterConfig, next_id: Callable[[], int]) -> None:
        self.cfg = config
        self._next_id = next_id
        self._pre_roll_len = _frames(config.pre_roll_ms) + config.start_frames
        self._end_frames = max(1, _frames(config.end_silence_ms))
        self._max_frames = _frames(config.max_utterance_s * 1000)
        self._cut_window = max(1, _frames(config.force_cut_window_s * 1000))
        self._min_speech_frames = _frames(config.min_utterance_ms)
        self._keep_tail = _frames(config.trailing_silence_keep_ms)
        self._partial_every = _frames(config.partial_interval_ms) if config.partial_interval_ms else 0
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._pre_roll: deque[tuple[AudioArray, float, float]] = deque(maxlen=self._pre_roll_len)
        self._run = 0  # consecutive frames >= start threshold while idle
        self._active_id: int | None = None
        self._frames: list[AudioArray] = []
        self._probs: list[float] = []
        self._t0 = 0.0  # start time of the first frame in _frames
        self._silence = 0  # consecutive frames < end threshold while active
        self._since_partial = 0
        self._partial_version = 0

    @property
    def in_speech(self) -> bool:
        return self._active_id is not None

    # ------------------------------------------------------------------ input
    def push(self, frame: AudioArray, prob: float, t: float) -> list[SegmenterOutput]:
        """Feed one 512-sample frame that starts at time ``t`` (seconds)."""
        if frame.shape[0] != FRAME_SAMPLES:
            raise ValueError(f"frame must have {FRAME_SAMPLES} samples, got {frame.shape[0]}")
        if self._active_id is None:
            return self._push_idle(frame, prob, t)
        return self._push_active(frame, prob)

    def flush(self) -> list[SegmenterOutput]:
        """End of stream: finish the utterance in progress, if any."""
        if self._active_id is None:
            return []
        return self._finish(trailing_silence=self._silence)

    # ------------------------------------------------------------------ idle
    def _push_idle(self, frame: AudioArray, prob: float, t: float) -> list[SegmenterOutput]:
        self._pre_roll.append((frame, prob, t))
        self._run = self._run + 1 if prob >= self.cfg.start_threshold else 0
        if self._run < self.cfg.start_frames:
            return []
        # Start: everything in the pre-roll (which ends with the triggering run) is speech
        # context for this utterance.
        self._active_id = self._next_id()
        self._frames = [f for f, _, _ in self._pre_roll]
        self._probs = [p for _, p, _ in self._pre_roll]
        self._t0 = self._pre_roll[0][2]
        self._pre_roll.clear()
        self._run = 0
        self._silence = 0
        self._since_partial = 0
        self._partial_version = 0
        return [SpeechStarted(self._active_id, self._t0)]

    # ------------------------------------------------------------------ active
    def _push_active(self, frame: AudioArray, prob: float) -> list[SegmenterOutput]:
        self._frames.append(frame)
        self._probs.append(prob)
        self._silence = self._silence + 1 if prob < self.cfg.end_threshold else 0

        if self._silence >= self._end_frames:
            return self._finish(trailing_silence=self._silence)

        if len(self._frames) >= self._max_frames:
            return self._force_cut()

        if self._partial_every:
            self._since_partial += 1
            # Only while speech is ongoing: a partial during trailing silence would just
            # repeat what the imminent final will say.
            if self._since_partial >= self._partial_every and self._silence == 0:
                self._since_partial = 0
                return [SegmentReady(self._segment(self._frames, is_final=False))]
        return []

    def _finish(self, trailing_silence: int) -> list[SegmenterOutput]:
        assert self._active_id is not None
        uid = self._active_id
        drop = max(0, trailing_silence - self._keep_tail)
        frames = self._frames[: len(self._frames) - drop] if drop else self._frames
        # Count frames that look like speech; pre-roll and pauses do not count.
        speech_frames = sum(1 for p in self._probs if p >= self.cfg.end_threshold)
        out: list[SegmenterOutput]
        if speech_frames < self._min_speech_frames:
            out = [Discarded(uid)]
        else:
            out = [SegmentReady(self._segment(frames, is_final=True))]
        self._active_id = None
        self._frames, self._probs = [], []
        self._silence = 0
        self._pre_roll.clear()
        return out

    def _force_cut(self) -> list[SegmenterOutput]:
        assert self._active_id is not None
        window_start = len(self._probs) - self._cut_window
        window = self._probs[window_start:]
        cut = window_start + int(np.argmin(window))  # last frame kept in this utterance
        head, tail = self._frames[: cut + 1], self._frames[cut + 1 :]
        tail_probs = self._probs[cut + 1 :]
        out: list[SegmenterOutput] = [SegmentReady(self._segment(head, is_final=True))]
        t_tail = self._t0 + (cut + 1) * FRAME_SECONDS
        # The speaker is still talking: the remainder starts the next utterance at once.
        self._active_id = self._next_id()
        self._frames, self._probs = list(tail), list(tail_probs)
        self._t0 = t_tail
        self._silence = 0
        for p in reversed(tail_probs):
            if p < self.cfg.end_threshold:
                self._silence += 1
            else:
                break
        self._since_partial = 0
        self._partial_version = 0
        out.append(SpeechStarted(self._active_id, t_tail))
        return out

    def _segment(self, frames: list[AudioArray], *, is_final: bool) -> SpeechSegment:
        assert self._active_id is not None
        audio = np.concatenate(frames).astype(np.float32, copy=False)
        version = self._partial_version
        if not is_final:
            self._partial_version += 1
        return SpeechSegment(
            utterance_id=self._active_id,
            audio=audio,
            t_start=self._t0,
            t_end=self._t0 + len(frames) * FRAME_SECONDS,
            is_final=is_final,
            version=version,
        )
