"""Windows: capture what the computer is playing (WASAPI loopback) via PyAudioWPatch.

No virtual audio cable is needed. Follows the default output device: when it changes
(headphones plugged in), the stream is reopened on the new device's loopback.

Note: WASAPI loopback delivers no data while nothing is playing. The pipeline's front end
fills such gaps with silence so utterances still end (``PipelineConfig.gap_fill``).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from polyglot.core.errors import AudioSourceError
from polyglot.core.ports import AudioSink

log = logging.getLogger(__name__)


def _pyaudio() -> Any:
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:
        raise AudioSourceError("PyAudioWPatch is required for loopback capture on Windows") from exc
    return pyaudio


def find_default_loopback(pa: Any, pyaudio: Any) -> dict[str, Any]:
    """The loopback twin of the current default output device."""
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError as exc:
        raise AudioSourceError("WASAPI is not available on this system") from exc
    speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
    if speakers.get("isLoopbackDevice"):
        return dict(speakers)
    for loopback in pa.get_loopback_device_info_generator():
        if speakers["name"] in loopback["name"]:
            return dict(loopback)
    raise AudioSourceError(f"no loopback device found for {speakers['name']!r}")


def list_loopback_devices() -> list[dict[str, Any]]:
    pyaudio = _pyaudio()
    pa = pyaudio.PyAudio()
    try:
        return [
            {"id": int(d["index"]), "name": str(d["name"]), "rate": int(d["defaultSampleRate"])}
            for d in pa.get_loopback_device_info_generator()
        ]
    finally:
        pa.terminate()


class WasapiLoopbackSource:
    def __init__(self, *, block_ms: int = 20, follow_default: bool = True) -> None:
        self._pyaudio = _pyaudio()
        self._pa = self._pyaudio.PyAudio()
        self._device = find_default_loopback(self._pa, self._pyaudio)
        self._rate = int(self._device["defaultSampleRate"])
        self._channels = max(1, int(self._device["maxInputChannels"]))
        self._block_ms = block_ms
        self._follow = follow_default
        self._stream: Any = None
        self._sink: AudioSink | None = None
        self._watch_stop = threading.Event()
        self._watcher: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._rate

    @property
    def device_name(self) -> str:
        return str(self._device["name"])

    def start(self, sink: AudioSink) -> None:
        self._sink = sink
        self._open()
        if self._follow:
            self._watch_stop.clear()
            self._watcher = threading.Thread(
                target=self._watch, name="polyglot-loopback-watch", daemon=True
            )
            self._watcher.start()

    def _open(self) -> None:
        pyaudio, sink, channels = self._pyaudio, self._sink, self._channels
        assert sink is not None

        def callback(
            in_data: bytes, frame_count: int, time_info: Any, status: int
        ) -> tuple[None, int]:
            sink.on_audio(np.frombuffer(in_data, dtype=np.float32).reshape(-1, channels))
            return None, pyaudio.paContinue

        try:
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=self._rate,
                input=True,
                input_device_index=int(self._device["index"]),
                frames_per_buffer=max(1, int(self._rate * self._block_ms / 1000)),
                stream_callback=callback,
            )
        except OSError as exc:
            raise AudioSourceError(f"cannot open loopback on {self.device_name!r}: {exc}") from exc
        log.info("loopback capture on %s at %d Hz x%d", self.device_name, self._rate, channels)

    def _close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except OSError:
                log.debug("error closing loopback stream", exc_info=True)
            self._stream = None

    def _watch(self) -> None:
        while not self._watch_stop.wait(2.0):
            try:
                current = find_default_loopback(self._pa, self._pyaudio)
            except AudioSourceError:
                continue
            if current["name"] == self._device["name"]:
                continue
            with self._lock:
                if (
                    int(current["defaultSampleRate"]) != self._rate
                    or int(current["maxInputChannels"]) != self._channels
                ):
                    # The pipeline's buffers were sized for the old format: ask for a restart.
                    if self._sink is not None:
                        self._sink.on_error(
                            AudioSourceError(
                                f"default output changed to {current['name']!r} with a different "
                                "format; press Start to listen again"
                            )
                        )
                    return
                log.info("default output changed to %s; following it", current["name"])
                self._close()
                self._device = current
                self._open()

    def stop(self) -> None:
        self._watch_stop.set()
        if self._watcher is not None and self._watcher is not threading.current_thread():
            self._watcher.join(timeout=3.0)
        with self._lock:
            self._close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self._pa.terminate()
        except Exception:
            pass
