"""Resampler adapters: device rate -> 16 kHz, streaming (state kept across blocks)."""

from __future__ import annotations

import numpy as np

from polyglot.core.models import SAMPLE_RATE
from polyglot.core.ports import RawAudio, Resampler


class IdentityResampler:
    """For sources that already deliver 16 kHz."""

    def process(self, block: RawAudio) -> RawAudio:
        return np.asarray(block, dtype=np.float32).reshape(-1)

    def flush(self) -> RawAudio:
        return np.zeros(0, dtype=np.float32)


class SoxrResampler:
    """High-quality streaming resampler (libsoxr)."""

    def __init__(self, in_rate: int, out_rate: int = SAMPLE_RATE) -> None:
        import soxr  # lazy: native library

        self._stream = soxr.ResampleStream(in_rate, out_rate, 1, dtype="float32", quality="HQ")

    def process(self, block: RawAudio) -> RawAudio:
        data = np.ascontiguousarray(block, dtype=np.float32).reshape(-1)
        return np.asarray(self._stream.resample_chunk(data), dtype=np.float32)

    def flush(self) -> RawAudio:
        tail = self._stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
        return np.asarray(tail, dtype=np.float32)


def make_resampler(in_rate: int) -> Resampler:
    if in_rate == SAMPLE_RATE:
        return IdentityResampler()
    return SoxrResampler(in_rate)
