"""Silero VAD run directly with onnxruntime.

By default the model bundled with faster-whisper is used, so no extra download is needed.
Both published single-model interfaces are supported:

* v5: inputs ``input`` (1, 64+512), ``state`` (2, 1, 128), ``sr``; 64 samples of context
  from the previous frame are prepended, as the reference implementation does.
* v4: inputs ``input`` (1, 512), ``h``/``c`` (2, 1, 64), ``sr``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from polyglot.core.errors import ConfigError
from polyglot.core.models import FRAME_SAMPLES, SAMPLE_RATE
from polyglot.core.ports import RawAudio

log = logging.getLogger(__name__)

_CONTEXT = 64


def bundled_model_candidates() -> list[Path]:
    try:
        import faster_whisper
    except ImportError:
        return []
    assets = Path(faster_whisper.__file__).parent / "assets"
    return sorted(assets.glob("silero*.onnx"))


class SileroVad:
    def __init__(self, model_path: str | Path | None = None, *, threads: int = 1) -> None:
        import onnxruntime as ort  # lazy: native library

        candidates = [Path(model_path)] if model_path else bundled_model_candidates()
        if not candidates:
            raise ConfigError("no Silero VAD model found (is faster-whisper installed?)")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = threads
        errors: list[str] = []
        for path in candidates:
            try:
                sess = ort.InferenceSession(
                    str(path), sess_options=opts, providers=["CPUExecutionProvider"]
                )
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            mode = self._detect(sess)
            if mode is not None:
                self._sess, self._mode, self.model_path = sess, mode, path
                break
            errors.append(f"{path.name}: unsupported inputs {[i.name for i in sess.get_inputs()]}")
        else:
            raise ConfigError("no usable Silero VAD model: " + "; ".join(errors))
        self._input_names = {i.name for i in self._sess.get_inputs()}
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        log.info("Silero VAD %s from %s", self._mode, self.model_path)
        self.reset()

    @staticmethod
    def _detect(sess: Any) -> str | None:
        names = {i.name for i in sess.get_inputs()}
        if {"input", "state"} <= names:
            return "v5"
        if {"input", "h", "c"} <= names:
            return "v4"
        return None

    def reset(self) -> None:
        if self._mode == "v5":
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            self._context = np.zeros((1, _CONTEXT), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def speech_prob(self, frame: RawAudio) -> float:
        if frame.shape[0] != FRAME_SAMPLES:
            raise ValueError(f"Silero expects {FRAME_SAMPLES}-sample frames")
        x = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        if self._mode == "v5":
            x = np.concatenate((self._context, x), axis=1)
            feeds: dict[str, Any] = {"input": x, "state": self._state}
            if "sr" in self._input_names:
                feeds["sr"] = self._sr
            out, self._state = self._sess.run(None, feeds)[:2]
            self._context = x[:, -_CONTEXT:]
        else:
            feeds = {"input": x, "h": self._h, "c": self._c}
            if "sr" in self._input_names:
                feeds["sr"] = self._sr
            out, self._h, self._c = self._sess.run(None, feeds)[:3]
        return float(np.asarray(out).reshape(-1)[0])
