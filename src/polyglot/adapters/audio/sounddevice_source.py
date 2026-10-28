"""Any input device through PortAudio (sounddevice): microphones, BlackHole on macOS,
VB-Cable/Voicemeeter on Windows."""

from __future__ import annotations

import logging
from typing import Any

from polyglot.core.errors import AudioSourceError
from polyglot.core.ports import AudioSink

log = logging.getLogger(__name__)


def _sd() -> Any:
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio missing
        raise AudioSourceError(f"PortAudio is not available: {exc}") from exc
    return sd


class SoundDeviceSource:
    def __init__(self, device: int | str | None = None, *, block_ms: int = 20) -> None:
        sd = _sd()
        try:
            info = sd.query_devices(device, kind="input")
        except (ValueError, sd.PortAudioError) as exc:
            raise AudioSourceError(f"input device {device!r} not found: {exc}") from exc
        self.device = device
        self.device_name: str = str(info["name"])
        self._rate = int(info["default_samplerate"])
        self._channels = max(1, min(2, int(info["max_input_channels"])))
        self._block = max(1, int(self._rate * block_ms / 1000))
        self._stream: Any = None
        self._stopping = False

    @property
    def sample_rate(self) -> int:
        return self._rate

    def start(self, sink: AudioSink) -> None:
        sd = _sd()
        self._stopping = False

        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            if status:
                log.debug("input status: %s", status)
            sink.on_audio(indata)  # the sink copies synchronously

        def finished() -> None:
            if not self._stopping:
                sink.on_error(AudioSourceError(f"input stream from {self.device_name!r} stopped"))

        try:
            self._stream = sd.InputStream(
                device=self.device,
                channels=self._channels,
                samplerate=self._rate,
                dtype="float32",
                blocksize=self._block,
                callback=callback,
                finished_callback=finished,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            raise AudioSourceError(f"cannot open {self.device_name!r}: {exc}") from exc
        log.info("listening on %s at %d Hz x%d", self.device_name, self._rate, self._channels)

    def stop(self) -> None:
        self._stopping = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None


def list_input_devices() -> list[dict[str, Any]]:
    sd = _sd()
    out: list[dict[str, Any]] = []
    for idx, dev in enumerate(sd.query_devices()):
        if int(dev["max_input_channels"]) > 0:
            name = str(dev["name"])
            out.append(
                {
                    "id": idx,
                    "name": name,
                    "rate": int(dev["default_samplerate"]),
                    "virtual": any(
                        k in name.lower()
                        for k in (
                            "blackhole",
                            "vb-audio",
                            "cable",
                            "voicemeeter",
                            "loopback",
                            "soundflower",
                        )
                    ),
                }
            )
    return out
