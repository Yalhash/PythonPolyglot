"""The Parakeet and Qwen3-ASR adapters, against stand-ins for their libraries.

The real packages are optional (``--extra parakeet`` / ``--extra qwen3``); these tests
inject fake modules, so they run without them. ``tests/integration`` runs the real models.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from polyglot.adapters.asr import parakeet, qwen3
from polyglot.adapters.asr.parakeet import ParakeetTranscriber, choose_providers
from polyglot.adapters.asr.qwen3 import Qwen3Transcriber, language_code, language_name
from polyglot.config import AppConfig
from polyglot.core.errors import ConfigError, TranscriptionFailed
from polyglot.wiring import build_transcriber

AUDIO = np.zeros(16_000, dtype=np.float32)
CPU, CUDA, DML = "CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"


# ------------------------------------------------------------------ Parakeet
class FakeOnnxModel:
    def __init__(self, text: str | Exception = " Bonjour à tous. ") -> None:
        self.text = text
        self.calls: list[tuple[Any, int]] = []

    def recognize(self, waveform: Any, *, sample_rate: int) -> str:
        self.calls.append((waveform, sample_rate))
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


@pytest.fixture
def onnx(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Fake ``onnx_asr`` and ``onnxruntime``; returns what ``load_model`` recorded."""
    state = types.SimpleNamespace(loaded=[], model=FakeOnnxModel(), providers=[CPU])

    def load_model(name: str, path: Any = None, **kwargs: Any) -> FakeOnnxModel:
        state.loaded.append((name, path, kwargs))
        return state.model

    monkeypatch.setitem(sys.modules, "onnx_asr", types.SimpleNamespace(load_model=load_model))
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(get_available_providers=lambda: list(state.providers)),
    )
    return state


def test_provider_choice_never_needs_nvidia() -> None:
    assert choose_providers("auto", [CPU]) == [CPU]
    assert choose_providers("auto", [DML, CPU]) == [DML, CPU]  # AMD/Intel GPU via DirectML
    assert choose_providers("auto", [CUDA, DML, CPU]) == [CUDA, CPU]
    assert choose_providers("cpu", [CUDA, CPU]) == [CPU]
    assert choose_providers("cuda", [CUDA, CPU]) == [CUDA, CPU]
    with pytest.raises(ConfigError, match="onnxruntime-gpu"):
        choose_providers("cuda", [DML, CPU])


def test_parakeet_loads_int8_on_the_cpu_and_transcribes(onnx: Any) -> None:
    tr = ParakeetTranscriber()
    tr.load()
    tr.load()  # loading twice is a no-op
    assert len(onnx.loaded) == 1
    name, path, kwargs = onnx.loaded[0]
    assert name == "nemo-parakeet-tdt-0.6b-v3" and path is None
    assert kwargs == {"quantization": "int8", "providers": [CPU]}
    assert tr.effective_device == "cpu"
    t = tr.transcribe(AUDIO, None)
    assert t.text == "Bonjour à tous."
    assert t.language is None and t.language_probability == 0.0  # Parakeet does not say
    assert onnx.model.calls[0][1] == 16_000


def test_parakeet_reports_the_language_it_was_given(onnx: Any) -> None:
    tr = ParakeetTranscriber(quantization=None)
    tr.load()
    assert onnx.loaded[0][2]["quantization"] is None
    t = tr.transcribe(AUDIO, "fr")
    assert t.language == "fr" and t.language_probability == 1.0


def test_parakeet_uses_directml_when_installed(onnx: Any) -> None:
    onnx.providers = [DML, CPU]
    tr = ParakeetTranscriber()
    tr.load()
    assert onnx.loaded[0][2]["providers"] == [DML, CPU] and tr.effective_device == "dml"


def test_parakeet_errors_are_typed(onnx: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    tr = ParakeetTranscriber()
    with pytest.raises(TranscriptionFailed, match="not loaded"):
        tr.transcribe(AUDIO, None)
    tr.load()
    onnx.model.text = RuntimeError("boom")
    with pytest.raises(TranscriptionFailed, match="boom"):
        tr.transcribe(AUDIO, None)

    def broken(*_a: Any, **_k: Any) -> None:
        raise OSError("no network")

    monkeypatch.setattr(sys.modules["onnx_asr"], "load_model", broken)
    with pytest.raises(TranscriptionFailed, match="no network"):
        ParakeetTranscriber().load()


def test_parakeet_not_installed_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnx_asr", None)  # import raises ImportError
    with pytest.raises(ConfigError, match="--extra parakeet"):
        ParakeetTranscriber().load()


def test_parakeet_language_list_includes_french() -> None:
    assert "fr" in parakeet.LANGUAGES and len(parakeet.LANGUAGES) == 25


# ------------------------------------------------------------------ Qwen3-ASR
class FakeQwenModel:
    def __init__(self) -> None:
        self.result: Any = types.SimpleNamespace(language="French", text=" Bonjour. ")
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return [self.result]


@pytest.fixture
def qwen(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Fake ``torch`` and ``qwen_asr``; returns what ``from_pretrained`` recorded."""
    state = types.SimpleNamespace(loaded=[], model=FakeQwenModel(), cuda=False)

    class Qwen3ASRModel:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> FakeQwenModel:
            state.loaded.append((name, kwargs))
            return state.model

    torch = types.SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        cuda=types.SimpleNamespace(is_available=lambda: state.cuda),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", types.SimpleNamespace(Qwen3ASRModel=Qwen3ASRModel))
    return state


def test_qwen3_runs_on_the_cpu_in_float32_without_a_gpu(qwen: Any) -> None:
    tr = Qwen3Transcriber()
    tr.load()
    name, kwargs = qwen.loaded[0]
    assert name == "Qwen/Qwen3-ASR-0.6B"
    assert kwargs["device_map"] == "cpu" and kwargs["dtype"] == "fp32"
    assert tr.effective_device == "cpu"


def test_qwen3_uses_a_cuda_gpu_unless_told_not_to(qwen: Any) -> None:
    qwen.cuda = True
    Qwen3Transcriber().load()
    assert qwen.loaded[0][1]["device_map"] == "cuda:0" and qwen.loaded[0][1]["dtype"] == "bf16"
    Qwen3Transcriber(device="cpu").load()
    assert qwen.loaded[1][1]["device_map"] == "cpu"


def test_qwen3_cuda_without_a_gpu_is_a_config_error(qwen: Any) -> None:
    with pytest.raises(ConfigError, match="CUDA"):
        Qwen3Transcriber(device="cuda").load()


def test_qwen3_detects_the_language(qwen: Any) -> None:
    tr = Qwen3Transcriber()
    tr.load()
    t = tr.transcribe(AUDIO, None)
    assert t.text == "Bonjour." and t.language == "fr" and t.language_probability == 1.0
    audio, rate = qwen.model.calls[0]["audio"]
    assert rate == 16_000 and audio.dtype == np.float32
    assert qwen.model.calls[0]["language"] is None  # auto-detect


def test_qwen3_forces_a_locked_language_by_name(qwen: Any) -> None:
    tr = Qwen3Transcriber()
    tr.load()
    qwen.model.result = types.SimpleNamespace(language="", text="Hola")
    t = tr.transcribe(AUDIO, "es")
    assert qwen.model.calls[0]["language"] == "Spanish"
    assert t.language == "es"
    tr.transcribe(AUDIO, "cy")  # Welsh: not one of Qwen's, so it auto-detects
    assert qwen.model.calls[1]["language"] is None


def test_qwen3_silence_has_no_language(qwen: Any) -> None:
    tr = Qwen3Transcriber()
    tr.load()
    qwen.model.result = types.SimpleNamespace(language="English", text="  ")
    t = tr.transcribe(AUDIO, None)
    assert t.text == "" and t.language is None and t.language_probability == 0.0


def test_qwen3_errors_are_typed(qwen: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    tr = Qwen3Transcriber()
    with pytest.raises(TranscriptionFailed, match="not loaded"):
        tr.transcribe(AUDIO, None)
    tr.load()
    qwen.model.result = RuntimeError("out of memory")
    with pytest.raises(TranscriptionFailed, match="out of memory"):
        tr.transcribe(AUDIO, None)
    monkeypatch.setitem(sys.modules, "qwen_asr", None)
    with pytest.raises(ConfigError, match="--extra qwen3"):
        Qwen3Transcriber().load()


def test_qwen3_language_names_and_codes() -> None:
    assert len(qwen3.LANGUAGES) == 30
    assert language_name("fr") == "French" and language_name("pt-BR") == "Portuguese"
    assert language_name("fil") == "Filipino" and language_name(None) is None
    assert language_code("French") == "fr" and language_code("fr") == "fr"
    assert language_code("Filipino") == "tl" and language_code("Klingon") is None


# ------------------------------------------------------------------ wiring
@pytest.mark.parametrize(
    ("engine", "cls"),
    [("whisper", "FasterWhisperTranscriber"), ("parakeet", "ParakeetTranscriber"),
     ("qwen3", "Qwen3Transcriber")],
)  # fmt: skip
def test_build_transcriber_follows_the_engine_setting(engine: str, cls: str) -> None:
    cfg = AppConfig()
    cfg.asr.engine = engine  # type: ignore[assignment]
    tr = build_transcriber(cfg)
    assert type(tr).__name__ == cls
    assert getattr(tr, "_model", None) is None  # nothing heavy loads until load()


def test_build_transcriber_passes_engine_settings() -> None:
    cfg = AppConfig()
    cfg.asr.engine, cfg.asr.device = "parakeet", "cpu"
    cfg.asr.parakeet_quantization = "none"
    p = build_transcriber(cfg)
    assert isinstance(p, ParakeetTranscriber)
    assert p.quantization is None and p.device == "cpu"
    cfg.asr.engine, cfg.asr.qwen3_model = "qwen3", "Qwen/Qwen3-ASR-1.7B"
    q = build_transcriber(cfg)
    assert isinstance(q, Qwen3Transcriber) and q.model_name == "Qwen/Qwen3-ASR-1.7B"
