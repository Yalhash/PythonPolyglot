"""Sources that play back recorded audio: a WAV file, or an in-memory array.

Used for tests, for the headless CLI, and for replaying recorded sessions while tuning
the segmenter. ``realtime=True`` paces blocks like a live device; ``False`` delivers as
fast as possible (size the pipeline's ring buffer to the recording when doing that).
"""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path

import numpy as np

from polyglot.core.errors import AudioSourceError
from polyglot.core.ports import AudioSink, RawAudio


class ArraySource:
    def __init__(
        self,
        samples: RawAudio,
        sample_rate: int,
        *,
        realtime: bool = False,
        block_ms: int = 100,
    ) -> None:
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim not in (1, 2):
            raise ValueError("samples must be (frames,) or (frames, channels)")
        self._data = data
        self._rate = sample_rate
        self._realtime = realtime
        self._block = max(1, int(sample_rate * block_ms / 1000))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def sample_rate(self) -> int:
        return self._rate

    @property
    def duration_s(self) -> float:
        return float(self._data.shape[0] / self._rate)

    def start(self, sink: AudioSink) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._play, args=(sink,), name="polyglot-playback", daemon=True
        )
        self._thread.start()

    def _play(self, sink: AudioSink) -> None:
        period = self._block / self._rate
        next_t = time.monotonic()
        for i in range(0, self._data.shape[0], self._block):
            if self._stop.is_set():
                return
            sink.on_audio(self._data[i : i + self._block])
            if self._realtime:
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        if not self._stop.is_set():
            sink.on_end()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)


def read_wav(path: str | Path) -> tuple[RawAudio, int]:
    """Read a PCM WAV file as float32 in [-1, 1]; shape (frames,) or (frames, channels)."""
    try:
        with wave.open(str(path), "rb") as wf:
            channels, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    except (OSError, wave.Error) as exc:
        raise AudioSourceError(f"cannot read WAV file {path}: {exc}") from exc
    # Truncate a partial last frame (a cut-off recording) instead of failing.
    frame_bytes = width * channels
    raw = raw[: len(raw) - len(raw) % frame_bytes] if frame_bytes else raw
    try:
        if width == 1:
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 2:
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 3:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            ints = (
                b[:, 0].astype(np.int32)
                | (b[:, 1].astype(np.int32) << 8)
                | (b[:, 2].astype(np.int32) << 16)
            )
            ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
            data = ints.astype(np.float32) / 8388608.0
        elif width == 4:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:  # pragma: no cover - wave module rejects other widths
            raise AudioSourceError(f"unsupported sample width {width}")
        if channels > 1:
            data = data.reshape(-1, channels)
    except ValueError as exc:
        raise AudioSourceError(f"cannot decode WAV file {path}: {exc}") from exc
    return data, rate


def write_wav(path: str | Path, samples: RawAudio, sample_rate: int) -> None:
    """Write mono float32 samples as 16-bit PCM."""
    scaled = np.round(np.asarray(samples, dtype=np.float64) * 32768.0)  # inverse of read_wav
    pcm = np.clip(scaled, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class WavFileSource(ArraySource):
    def __init__(self, path: str | Path, *, realtime: bool = False, block_ms: int = 100) -> None:
        data, rate = read_wav(path)
        super().__init__(data, rate, realtime=realtime, block_ms=block_ms)
        self.path = Path(path)
