"""Real models on synthesized speech. Slow: run with ``pytest -m slow``.

Needs faster-whisper (downloads the ``tiny`` model on first run, ~75 MB) and espeak-ng
(or espeak) to synthesize speech. CI runs these in the "models" job.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from polyglot.adapters.audio.playback import WavFileSource, read_wav
from polyglot.adapters.audio.resample import make_resampler
from polyglot.core.events import EventBus
from polyglot.core.models import FRAME_SAMPLES
from polyglot.core.pipeline import Pipeline, PipelineConfig
from tests.conftest import Recorder, wait_for

pytestmark = pytest.mark.slow

SENTENCE = "Hello everyone. This is a test of the transcription system."


@pytest.fixture(scope="module")
def speech_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe is None:
        pytest.skip("espeak-ng is not installed")
    out = tmp_path_factory.mktemp("speech") / "speech.wav"
    subprocess.run([exe, "-w", str(out), SENTENCE], check=True, capture_output=True)  # noqa: S603
    # Pad with silence so the utterance can end before the file does.
    data, rate = read_wav(out)
    from polyglot.adapters.audio.playback import write_wav

    padded = np.concatenate((np.zeros(rate // 2, np.float32), data, np.zeros(rate, np.float32)))
    write_wav(out, padded, rate)
    return out


def test_silero_detects_synthesized_speech(speech_wav: Path) -> None:
    pytest.importorskip("faster_whisper")
    pytest.importorskip("soxr")
    from polyglot.adapters.vad.silero import SileroVad

    data, rate = read_wav(speech_wav)
    r = make_resampler(rate)
    audio = np.concatenate((r.process(data), r.flush()))
    vad = SileroVad()
    probs = [vad.speech_prob(audio[i : i + FRAME_SAMPLES]) for i in range(0, audio.size - FRAME_SAMPLES, FRAME_SAMPLES)]
    assert max(probs) > 0.5
    assert min(probs[:5]) < 0.2  # the leading silence


def test_whisper_tiny_end_to_end(speech_wav: Path) -> None:
    pytest.importorskip("faster_whisper")
    pytest.importorskip("soxr")
    from polyglot.adapters.asr.faster_whisper import FasterWhisperTranscriber
    from polyglot.adapters.vad.silero import SileroVad

    bus = EventBus()
    rec = Recorder(bus)
    p = Pipeline(
        source=WavFileSource(speech_wav, realtime=False),
        resampler_factory=make_resampler,
        vad=SileroVad(),
        transcriber=FasterWhisperTranscriber("tiny", device="cpu", compute_type="int8"),
        translator=None,
        suggester=None,
        bus=bus,
        config=PipelineConfig(ring_seconds=120),
    )
    try:
        p.start()
        assert wait_for(lambda: p.source_ended, 120)
        assert p.wait_idle(120)
        text = " ".join(u.text for u in rec.finals()).lower()
        assert "test" in text and "transcription" in text, text
        assert all(u.source_lang == "en" for u in rec.finals())
    finally:
        p.close()
