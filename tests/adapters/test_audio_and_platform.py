"""Audio adapters, VAD, platform shield, secrets and dumps."""

from __future__ import annotations

import sys
import threading
import types
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from polyglot.adapters.audio.playback import ArraySource, WavFileSource, read_wav, write_wav
from polyglot.adapters.audio.resample import IdentityResampler, make_resampler
from polyglot.adapters.dump import SegmentDumper
from polyglot.adapters.fakes import NoopShield, ScriptedVad, silence, speech_like, tone
from polyglot.adapters.secrets import KeyringSecretStore, MemorySecretStore
from polyglot.adapters.vad.energy import EnergyVad
from polyglot.core.errors import AudioSourceError
from polyglot.core.models import FRAME_SAMPLES, SAMPLE_RATE, SpeechSegment
from polyglot.core.ports import ShieldStatus
from polyglot.platform import UnsupportedShield, current, window_shield
from polyglot.platform.windows import MIN_BUILD, WDA_EXCLUDEFROMCAPTURE, WindowsShield


class Sink:
    def __init__(self) -> None:
        self.blocks: list[np.ndarray] = []
        self.ended = threading.Event()
        self.errors: list[Exception] = []

    def on_audio(self, block: np.ndarray) -> None:
        self.blocks.append(np.array(block))

    def on_end(self) -> None:
        self.ended.set()

    def on_error(self, error: Exception) -> None:
        self.errors.append(error)


# ------------------------------------------------------------------ WAV / playback
def test_wav_round_trip_16bit(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    data = tone(0.5, amp=0.5)
    write_wav(path, data, SAMPLE_RATE)
    back, rate = read_wav(path)
    assert rate == SAMPLE_RATE and back.dtype == np.float32
    np.testing.assert_allclose(back, data, atol=1 / 32767 + 1e-6)


@pytest.mark.parametrize(("width", "dtype", "scale", "offset"), [(1, np.uint8, 127, 128), (4, "<i4", 2**31 - 1, 0)])
def test_wav_other_widths_and_stereo(tmp_path: Path, width: int, dtype: Any, scale: int, offset: int) -> None:
    path = tmp_path / "s.wav"
    frames = np.array([[0.5, -0.5], [0.25, 0.0]], dtype=np.float64)
    ints = (frames * scale + offset).astype(dtype)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(width)
        wf.setframerate(8000)
        wf.writeframes(ints.tobytes())
    data, rate = read_wav(path)
    assert rate == 8000 and data.shape == (2, 2)
    np.testing.assert_allclose(data, frames, atol=0.02)


def test_wav_24bit(tmp_path: Path) -> None:
    path = tmp_path / "24.wav"
    vals = np.array([0.5, -0.5, 0.0], dtype=np.float64)
    ints = (vals * (2**23 - 1)).astype(np.int32)
    raw = b"".join(int(v).to_bytes(4, "little", signed=True)[:3] for v in ints)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)
        wf.setframerate(16000)
        wf.writeframes(raw)
    data, _ = read_wav(path)
    np.testing.assert_allclose(data, vals, atol=1e-4)


def test_unreadable_wav_is_audio_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav")
    with pytest.raises(AudioSourceError):
        read_wav(bad)
    with pytest.raises(AudioSourceError):
        WavFileSource(tmp_path / "missing.wav")


def test_array_source_delivers_everything_then_ends() -> None:
    data = np.arange(16000, dtype=np.float32)
    src = ArraySource(data, 16000, block_ms=100)
    sink = Sink()
    src.start(sink)
    assert sink.ended.wait(2)
    np.testing.assert_array_equal(np.concatenate(sink.blocks), data)
    assert all(len(b) == 1600 for b in sink.blocks)
    assert src.duration_s == 1.0


def test_array_source_realtime_pacing_and_stop() -> None:
    src = ArraySource(np.zeros(16000 * 5, dtype=np.float32), 16000, realtime=True, block_ms=100)
    sink = Sink()
    src.start(sink)
    threading.Event().wait(0.25)
    src.stop()
    assert 1 <= len(sink.blocks) <= 6  # ~3 blocks in 250 ms, not all 50
    assert not sink.ended.is_set()


def test_wav_file_source(tmp_path: Path) -> None:
    path = tmp_path / "f.wav"
    write_wav(path, tone(0.2), 16000)
    src = WavFileSource(path)
    assert src.sample_rate == 16000 and src.path == path


def test_array_source_rejects_3d() -> None:
    with pytest.raises(ValueError):
        ArraySource(np.zeros((2, 2, 2), dtype=np.float32), 16000)


# ------------------------------------------------------------------ resampling
def test_identity_resampler_flattens() -> None:
    r = IdentityResampler()
    np.testing.assert_array_equal(r.process(np.ones((3, 1), dtype=np.float32)), np.ones(3))
    assert r.flush().size == 0
    assert isinstance(make_resampler(16000), IdentityResampler)


def test_soxr_resampler_streams_48k_to_16k() -> None:
    pytest.importorskip("soxr")
    r = make_resampler(48000)
    sig = tone(1.0, freq=440, rate=48000)
    out = np.concatenate([r.process(sig[i : i + 960]) for i in range(0, sig.size, 960)] + [r.flush()])
    assert abs(out.size - 16000) <= 16
    crossings = np.count_nonzero(np.diff(np.signbit(out[100:-100])))
    assert crossings == pytest.approx(2 * 440 * (out.size - 200) / 16000, rel=0.02)


# ------------------------------------------------------------------ VAD
def frames(audio: np.ndarray) -> list[np.ndarray]:
    return [audio[i : i + FRAME_SAMPLES] for i in range(0, audio.size - FRAME_SAMPLES + 1, FRAME_SAMPLES)]


def test_energy_vad_separates_tone_from_silence_and_adapts() -> None:
    vad = EnergyVad()
    noise = (np.random.default_rng(0).standard_normal(16000) * 0.003).astype(np.float32)
    quiet = [vad.speech_prob(f) for f in frames(noise)]
    loud = [vad.speech_prob(f) for f in frames(tone(0.5, amp=0.2))]
    assert max(quiet) < 0.5 < min(loud)
    vad.reset()
    assert vad.floor_db == vad.initial_floor_db
    assert vad.speech_prob(np.zeros(0, dtype=np.float32)) < 0.01


def test_scripted_vad() -> None:
    v = ScriptedVad([0.9, 0.1], default=0.5)
    f = silence(0.032)
    assert [v.speech_prob(f) for _ in range(3)] == [0.9, 0.1, 0.5]


def test_silero_vad_on_silence_and_tone() -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("faster_whisper")
    from polyglot.adapters.vad.silero import SileroVad

    vad = SileroVad()
    probs = [vad.speech_prob(f) for f in frames(silence(1.0))]
    assert max(probs) < 0.3
    vad.reset()
    with pytest.raises(ValueError):
        vad.speech_prob(np.zeros(100, dtype=np.float32))


def test_silero_reports_missing_model(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    from polyglot.adapters.vad.silero import SileroVad
    from polyglot.core.errors import ConfigError

    with pytest.raises(ConfigError):
        SileroVad(tmp_path / "nope.onnx")


# ------------------------------------------------------------------ Windows shield (fake user32)
class FakeUser32:
    def __init__(self, set_ok: bool = True, readback: int = WDA_EXCLUDEFROMCAPTURE) -> None:
        self.set_ok, self.readback = set_ok, readback
        self.set_calls: list[tuple[int, int]] = []

    def SetWindowDisplayAffinity(self, hwnd: int, affinity: int) -> int:
        self.set_calls.append((hwnd, affinity))
        return int(self.set_ok)

    def GetWindowDisplayAffinity(self, hwnd: int, ref: Any) -> int:
        ref._obj.value = self.readback
        return 1


def test_shield_hides_on_supported_build() -> None:
    api = FakeUser32()
    shield = WindowsShield(user32=api, build=MIN_BUILD, last_error=lambda: 0)
    assert shield.exclude_from_capture(0x1234) is ShieldStatus.HIDDEN
    assert api.set_calls == [(0x1234, WDA_EXCLUDEFROMCAPTURE)]


def test_shield_skips_old_builds_to_avoid_black_box() -> None:
    api = FakeUser32()
    assert WindowsShield(user32=api, build=MIN_BUILD - 1).exclude_from_capture(1) is ShieldStatus.UNSUPPORTED
    assert api.set_calls == []  # never called: WDA_MONITOR fallback would show a black box


def test_shield_reports_failure_and_bad_readback() -> None:
    assert WindowsShield(user32=FakeUser32(set_ok=False), build=MIN_BUILD, last_error=lambda: 5).exclude_from_capture(1) is ShieldStatus.FAILED
    assert WindowsShield(user32=FakeUser32(readback=1), build=MIN_BUILD, last_error=lambda: 0).exclude_from_capture(1) is ShieldStatus.FAILED


def test_platform_selection() -> None:
    info = current()
    shield = window_shield()
    if sys.platform == "win32":
        assert info.has_loopback and isinstance(shield, WindowsShield)
    else:
        assert not info.has_loopback
        assert isinstance(shield, UnsupportedShield)
        assert shield.exclude_from_capture(1) is ShieldStatus.UNSUPPORTED
    assert NoopShield(ShieldStatus.HIDDEN).exclude_from_capture(5) is ShieldStatus.HIDDEN


# ------------------------------------------------------------------ secrets
def test_memory_secret_store() -> None:
    s = MemorySecretStore({"a": "1"})
    assert s.get("a") == "1" and s.get("b") is None
    s.set("a", None)
    assert s.get("a") is None


def test_keyring_store_with_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    stored: dict[tuple[str, str], str] = {}

    class KeyringError(Exception):
        pass

    class PasswordDeleteError(KeyringError):
        pass

    fake = types.ModuleType("keyring")
    fake.get_password = lambda svc, name: stored.get((svc, name))  # type: ignore[attr-defined]
    fake.set_password = lambda svc, name, v: stored.__setitem__((svc, name), v)  # type: ignore[attr-defined]

    def delete(svc: str, name: str) -> None:
        if (svc, name) not in stored:
            raise PasswordDeleteError
        del stored[(svc, name)]

    fake.delete_password = delete  # type: ignore[attr-defined]
    errors = types.ModuleType("keyring.errors")
    errors.KeyringError = KeyringError  # type: ignore[attr-defined]
    errors.PasswordDeleteError = PasswordDeleteError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keyring", fake)
    monkeypatch.setitem(sys.modules, "keyring.errors", errors)

    store = KeyringSecretStore("polyglot-test")
    assert store.get("k") is None
    store.set("k", "secret")
    assert store.get("k") == "secret" and stored == {("polyglot-test", "k"): "secret"}
    store.set("k", None)
    store.set("k", None)  # deleting twice is fine
    assert store.get("k") is None


# ------------------------------------------------------------------ dumps / fakes
def test_segment_dumper_writes_wav(tmp_path: Path) -> None:
    dumper = SegmentDumper(tmp_path)
    dumper(SpeechSegment(7, tone(0.1), 0.0, 0.1, True))
    (path,) = list(dumper.folder.glob("*.wav"))
    assert path.name == "utt-00007.wav"
    data, rate = read_wav(path)
    assert rate == SAMPLE_RATE and data.size == 1600


def test_speech_like_pads_to_whole_frames() -> None:
    audio = speech_like([("tone", 0.1), ("silence", 0.05)])
    assert audio.size % FRAME_SAMPLES == 0
    assert audio.size >= int(0.15 * SAMPLE_RATE)
