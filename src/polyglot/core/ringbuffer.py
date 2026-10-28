"""Preallocated single-producer / single-consumer audio ring buffer.

The producer is the audio library's callback thread, so ``write`` does only a bounded
numpy copy into preallocated memory. When the consumer falls behind, the oldest audio is
overwritten and counted in ``overrun_samples`` rather than growing memory or blocking the
audio thread.
"""

from __future__ import annotations

import threading

import numpy as np

from polyglot.core.ports import RawAudio


class RingBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._read = 0  # absolute sample index of the next sample to read
        self._write = 0  # absolute sample index of the next sample to write
        self._lock = threading.Lock()
        self._data_ready = threading.Event()
        self.overrun_samples = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def available(self) -> int:
        with self._lock:
            return self._write - self._read

    def write(self, block: RawAudio) -> None:
        """Append mono samples. Never blocks for longer than one copy."""
        data = np.asarray(block, dtype=np.float32).reshape(-1)
        n = data.shape[0]
        if n == 0:
            return
        if n > self._capacity:  # keep only the newest `capacity` samples
            dropped = n - self._capacity
            data = data[dropped:]
            n = self._capacity
        else:
            dropped = 0
        with self._lock:
            start = self._write % self._capacity
            first = min(n, self._capacity - start)
            self._buf[start : start + first] = data[:first]
            if first < n:
                self._buf[: n - first] = data[first:]
            self._write += n
            # Samples skipped by an oversized block were never readable, and samples
            # overwritten before the reader got to them are lost: both are overruns.
            self._write_overrun_check()
            self.overrun_samples += dropped
        self._data_ready.set()

    def _write_overrun_check(self) -> None:
        lag = self._write - self._read
        if lag > self._capacity:
            lost = lag - self._capacity
            self._read += lost
            self.overrun_samples += lost

    def read(self, max_samples: int | None = None) -> RawAudio:
        """Return up to ``max_samples`` (default: all) available samples, oldest first."""
        with self._lock:
            n = self._write - self._read
            if max_samples is not None:
                n = min(n, max_samples)
            out = np.empty(n, dtype=np.float32)
            start = self._read % self._capacity
            first = min(n, self._capacity - start)
            out[:first] = self._buf[start : start + first]
            if first < n:
                out[first:] = self._buf[: n - first]
            self._read += n
            if self._read == self._write:
                self._data_ready.clear()
            return out

    def wait(self, timeout: float) -> bool:
        """Block until data may be available or ``timeout`` passes."""
        return self._data_ready.wait(timeout)

    def clear(self) -> None:
        with self._lock:
            self._read = self._write
            self._data_ready.clear()
