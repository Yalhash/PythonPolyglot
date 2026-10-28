from __future__ import annotations

import threading

import numpy as np
import pytest

from polyglot.core.ringbuffer import RingBuffer


def arange(a: int, b: int) -> np.ndarray:
    return np.arange(a, b, dtype=np.float32)


def test_reads_back_in_order() -> None:
    rb = RingBuffer(10)
    rb.write(arange(0, 4))
    rb.write(arange(4, 7))
    assert rb.available() == 7
    np.testing.assert_array_equal(rb.read(), arange(0, 7))
    assert rb.available() == 0
    assert rb.read().size == 0


def test_wraps_around_without_loss() -> None:
    rb = RingBuffer(8)
    rb.write(arange(0, 6))
    np.testing.assert_array_equal(rb.read(4), arange(0, 4))
    rb.write(arange(6, 12))  # crosses the end of the backing array
    np.testing.assert_array_equal(rb.read(), arange(4, 12))
    assert rb.overrun_samples == 0


def test_overrun_drops_oldest_and_counts() -> None:
    rb = RingBuffer(5)
    rb.write(arange(0, 4))
    rb.write(arange(4, 8))  # 3 samples too many
    assert rb.overrun_samples == 3
    np.testing.assert_array_equal(rb.read(), arange(3, 8))


def test_oversized_block_keeps_newest() -> None:
    rb = RingBuffer(4)
    rb.write(arange(0, 10))
    assert rb.overrun_samples == 6
    np.testing.assert_array_equal(rb.read(), arange(6, 10))


def test_accepts_2d_column_and_empty() -> None:
    rb = RingBuffer(4)
    rb.write(np.zeros((0,), dtype=np.float32))
    rb.write(arange(0, 3).reshape(-1, 1))
    np.testing.assert_array_equal(rb.read(), arange(0, 3))


def test_wait_wakes_on_write_and_clear() -> None:
    rb = RingBuffer(16)
    assert not rb.wait(0.01)
    threading.Timer(0.02, lambda: rb.write(arange(0, 2))).start()
    assert rb.wait(1.0)
    rb.clear()
    assert rb.available() == 0
    assert not rb.wait(0.01)


def test_rejects_bad_capacity() -> None:
    with pytest.raises(ValueError):
        RingBuffer(0)


def test_concurrent_producer_consumer_preserves_order() -> None:
    rb = RingBuffer(4096)
    total = 50_000
    got: list[np.ndarray] = []

    def produce() -> None:
        for start in range(0, total, 100):
            rb.write(arange(start, start + 100))

    t = threading.Thread(target=produce)
    t.start()
    n = 0
    while n < total and (t.is_alive() or rb.available()):
        chunk = rb.read()
        n += chunk.size
        got.append(chunk)
        rb.wait(0.001)
    t.join()
    got.append(rb.read())
    data = np.concatenate(got)
    # Whatever survived must be strictly increasing (no reordering, no duplication),
    # and everything lost must be accounted for as overrun.
    assert np.all(np.diff(data) > 0)
    assert data.size + rb.overrun_samples == total
