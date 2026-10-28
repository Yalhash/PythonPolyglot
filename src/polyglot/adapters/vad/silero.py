"""Silero VAD run directly with onnxruntime.

By default the model bundled with faster-whisper is used, so no extra download is needed.
Silero has shipped several ONNX interfaces, so the adapter reads the interface from the
model instead of assuming one:

* frame input ``input``: 512 new samples, plus 64 samples of context from the previous
  frame when the model expects 576 (faster-whisper's bundled model) or uses a single
  ``state`` tensor (Silero v5);
* recurrent state: every other input except ``sr`` (``state``, or ``h``/``c``), shaped
  from the model's declared input shapes and fed back from the matching outputs;
* ``sr`` (sample rate) only if the model declares it.
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


def _concrete(shape: Any) -> tuple[int, ...]:
    """Declared ONNX shape -> concrete shape; symbolic/unknown dimensions become 1."""
    return tuple(d if isinstance(d, int) and d > 0 else 1 for d in (shape or []))


class _Interface:
    def __init__(self, sess: Any) -> None:
        inputs = {i.name: i for i in sess.get_inputs()}
        if "input" not in inputs:
            raise ValueError(f"no 'input' tensor (inputs: {sorted(inputs)})")
        self.has_sr = "sr" in inputs
        self.state_names = [n for n in inputs if n not in ("input", "sr")]
        if not self.state_names:
            raise ValueError("no recurrent state inputs")
        self.state_shapes = {n: _concrete(inputs[n].shape) for n in self.state_names}
        self.state_dtypes = {n: _np_dtype(inputs[n].type) for n in self.state_names}
        shape = list(inputs["input"].shape or [])
        last = shape[-1] if shape else None
        if isinstance(last, int) and last > FRAME_SAMPLES:
            self.context = last - FRAME_SAMPLES
        elif isinstance(last, int) and last == FRAME_SAMPLES:
            self.context = 0
        else:  # dynamic length: v5 with a single "state" expects context; v4 does not
            self.context = _CONTEXT if self.state_names == ["state"] else 0
        outputs = [o.name for o in sess.get_outputs()]
        if len(outputs) < 1 + len(self.state_names):
            raise ValueError(f"expected {1 + len(self.state_names)} outputs, got {outputs}")
        self.rank = len(shape) or 2

    def describe(self) -> str:
        return f"states={self.state_names} context={self.context}"


def _np_dtype(onnx_type: str) -> Any:
    return np.float16 if "float16" in (onnx_type or "") else np.float32


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
                iface = _Interface(sess)
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            self._sess, self._iface, self.model_path = sess, iface, path
            break
        else:
            raise ConfigError("no usable Silero VAD model: " + "; ".join(errors))
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        log.info("Silero VAD from %s (%s)", self.model_path, self._iface.describe())
        self.reset()

    def reset(self) -> None:
        i = self._iface
        self._state = {
            n: np.zeros(i.state_shapes[n], dtype=i.state_dtypes[n]) for n in i.state_names
        }
        self._context = np.zeros((1, i.context), dtype=np.float32)

    def speech_prob(self, frame: RawAudio) -> float:
        if frame.shape[0] != FRAME_SAMPLES:
            raise ValueError(f"Silero expects {FRAME_SAMPLES}-sample frames")
        i = self._iface
        x = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        if i.context:
            x = np.concatenate((self._context, x), axis=1)
            self._context = x[:, -i.context :]
        feeds: dict[str, Any] = {
            "input": x if i.rank == 2 else x.reshape((1,) * (i.rank - 1) + (-1,))
        }
        feeds.update(self._state)
        if i.has_sr:
            feeds["sr"] = self._sr
        out = self._sess.run(None, feeds)
        for name, value in zip(i.state_names, out[1 : 1 + len(i.state_names)], strict=False):
            self._state[name] = value
        return float(np.asarray(out[0]).reshape(-1)[0])
