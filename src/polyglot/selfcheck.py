"""``polyglot --self-check``: prove that every native piece the app needs can load.

Meant for packaged builds (PyInstaller), where a missing hidden import, DLL or data file
only shows up when a user clicks something. Each check imports or opens one thing; none
needs a network, an audio device, an API key or a window. Exit code 0 means all required
checks passed; optional engines (Parakeet, Qwen3) are reported but never fail the run.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from polyglot import __version__


def _numpy() -> str:
    import numpy

    return numpy.__version__


def _soxr() -> str:
    import numpy as np
    import soxr

    out = soxr.resample(np.zeros(4800, dtype=np.float32), 48000, 16000)
    return f"{soxr.__version__}, resampled {len(out)} samples"


def _onnxruntime() -> str:
    import onnxruntime

    return f"{onnxruntime.__version__} ({', '.join(onnxruntime.get_available_providers())})"


def _silero() -> str:
    import onnxruntime

    from polyglot.adapters.vad.silero import bundled_model_candidates

    models = bundled_model_candidates()
    if not models:
        raise RuntimeError("no Silero VAD model found in faster_whisper/assets")
    onnxruntime.InferenceSession(str(models[0]), providers=["CPUExecutionProvider"])
    return models[0].name


def _faster_whisper() -> str:
    import ctranslate2
    import faster_whisper
    from faster_whisper import WhisperModel  # noqa: F401  (pulls in av, tokenizers)

    return f"{faster_whisper.__version__} (ctranslate2 {ctranslate2.__version__})"


def _huggingface_hub() -> str:
    import huggingface_hub

    return huggingface_hub.__version__


def _sounddevice() -> str:
    import sounddevice

    return sounddevice.get_portaudio_version()[1]


def _loopback() -> str:
    from polyglot import platform

    if not platform.current().has_loopback:
        return "not used on this OS"
    import pyaudiowpatch

    return f"PyAudioWPatch {getattr(pyaudiowpatch, '__version__', 'installed')}"


def _qt() -> str:
    from PyQt6.QtCore import QT_VERSION_STR
    from PyQt6.QtWidgets import QApplication  # noqa: F401

    from polyglot.ui.qt.overlay import load_stylesheet

    if not load_stylesheet().strip():
        raise RuntimeError("theme.qss is empty")
    return f"Qt {QT_VERSION_STR}, stylesheet loaded"


def _keyring() -> str:
    import keyring

    backend = keyring.get_keyring()
    name = f"{type(backend).__module__}.{type(backend).__name__}"
    if "fail" in name.lower():
        raise RuntimeError(f"no usable keychain backend ({name})")
    return name


def _hotkeys() -> str:
    from pynput import keyboard

    return keyboard.Listener.__module__


def _parakeet() -> str:
    import onnx_asr

    return getattr(onnx_asr, "__version__", "installed")


def _qwen3() -> str:
    import qwen_asr  # noqa: F401
    import torch

    return f"torch {torch.__version__}"


REQUIRED: list[tuple[str, Callable[[], str]]] = [
    ("numpy", _numpy),
    ("soxr (resampling)", _soxr),
    ("onnxruntime", _onnxruntime),
    ("Silero VAD model", _silero),
    ("faster-whisper", _faster_whisper),
    ("huggingface_hub (model downloads)", _huggingface_hub),
    ("sounddevice (PortAudio)", _sounddevice),
    ("system audio loopback", _loopback),
    ("Qt and theme", _qt),
    ("keychain", _keyring),
    ("global hotkeys", _hotkeys),
]

OPTIONAL: list[tuple[str, Callable[[], str]]] = [
    ("Parakeet engine", _parakeet),
    ("Qwen3-ASR engine", _qwen3),
]


def run(report: Path | None = None) -> int:
    """Run every check, write the report to ``report`` (or stdout), return the exit code."""
    lines = [f"polyglot {__version__} self-check (frozen={getattr(sys, 'frozen', False)})"]
    failed = 0
    for required, checks in ((True, REQUIRED), (False, OPTIONAL)):
        for name, check in checks:
            try:
                lines.append(f"  ok    {name}: {check()}")
            except Exception as exc:
                if required:
                    failed += 1
                    lines.append(f"  FAIL  {name}: {exc!r}")
                    lines.extend("        " + ln for ln in traceback.format_exc().splitlines())
                else:
                    lines.append(f"  --    {name}: not installed ({type(exc).__name__})")
    lines.append(
        "all required checks passed" if not failed else f"{failed} required check(s) failed"
    )
    text = "\n".join(lines) + "\n"
    if report is not None:
        report.write_text(text, encoding="utf-8")
    else:
        out: TextIO | None = sys.stdout
        if out is not None:
            out.write(text)
    return 1 if failed else 0
