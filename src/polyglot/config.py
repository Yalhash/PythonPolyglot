"""Typed configuration with one precedence order.

Lowest to highest: defaults -> ``config.toml`` in the user config directory ->
``POLYGLOT_<SECTION>__<FIELD>`` environment variables -> changes saved from the settings
dialog (written back to ``config.toml``).

API keys are not part of the config. They live in the OS keyring (see ``Secrets``);
``POLYGLOT_OPENAI_API_KEY`` / ``POLYGLOT_GOOGLE_API_KEY`` (or the conventional
``OPENAI_API_KEY``) override the keyring for development.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from polyglot.core.errors import ConfigError
from polyglot.core.ports import SecretStore

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 only
    import tomli as tomllib

ENV_PREFIX = "POLYGLOT_"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AudioConfig(_Section):
    #: auto = WASAPI loopback on Windows, else the default input device
    source: Literal["auto", "loopback", "device", "file"] = "auto"
    device: str | None = None  # input device name or index, for source = "device"
    file: str | None = None  # WAV path, for source = "file"
    ring_seconds: float = Field(30.0, ge=1.0, le=600.0)
    gap_fill: bool = True


class VadConfig(_Section):
    engine: Literal["silero", "energy"] = "silero"
    model_path: str | None = None
    start_threshold: float = Field(0.5, ge=0.0, le=1.0)
    end_threshold: float = Field(0.35, ge=0.0, le=1.0)
    start_frames: int = Field(3, ge=1, le=50)
    end_silence_ms: int = Field(600, ge=100, le=5000)
    pre_roll_ms: int = Field(300, ge=0, le=2000)
    max_utterance_s: float = Field(15.0, ge=3.0, le=60.0)
    min_utterance_ms: int = Field(300, ge=0, le=5000)
    partial_interval_ms: int = Field(0, ge=0, le=10000)

    @model_validator(mode="after")
    def _thresholds(self) -> VadConfig:
        if self.end_threshold > self.start_threshold:
            raise ValueError("vad.end_threshold must not exceed vad.start_threshold")
        return self


class AsrConfig(_Section):
    model: str = "base"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute_type: str = "default"
    beam_size: int = Field(5, ge=1, le=10)
    language: str | None = None  # force a source language; None = auto-detect then lock
    lock_after: int = Field(2, ge=1, le=10)
    lock_min_probability: float = Field(0.8, ge=0.0, le=1.0)
    queue_size: int = Field(8, ge=1, le=64)


class TranslationConfig(_Section):
    enabled: bool = False  # cloud calls are opt-in
    provider: Literal["openai", "google"] = "openai"
    target_language: str = "en"
    model: str = "gpt-4o-mini"
    timeout_s: float = Field(3.0, gt=0.0, le=30.0)


class SuggestionsConfig(_Section):
    enabled: bool = False  # cloud calls are opt-in
    model: str = "gpt-4.1-mini"
    timeout_s: float = Field(6.0, gt=0.0, le=60.0)
    conversation_type: Literal["interview", "meeting", "casual"] = "meeting"
    history: int = Field(6, ge=0, le=30)
    options: int = Field(2, ge=1, le=3)


class UiConfig(_Section):
    opacity: float = Field(0.92, ge=0.2, le=1.0)
    x: int = 100
    y: int = 100
    width: int = Field(420, ge=280)
    height: int = Field(640, ge=300)
    hide_from_capture: bool = True
    hotkey_suggest: str = "<ctrl>+<alt>+s"
    hotkey_toggle: str = "<ctrl>+<alt>+h"


class PrivacyConfig(_Section):
    log_content: bool = False  # transcript text in logs (DEBUG only)
    dump_audio: bool = False  # save each utterance as WAV in the data directory


class AppConfig(_Section):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    suggestions: SuggestionsConfig = Field(default_factory=SuggestionsConfig)
    ui: UiConfig = Field(default_factory=UiConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)


SECTIONS = tuple(AppConfig.model_fields)


# ----------------------------------------------------------------- paths
def config_dir() -> Path:
    import platformdirs

    return Path(platformdirs.user_config_dir("polyglot", appauthor=False))


def data_dir() -> Path:
    import platformdirs

    return Path(platformdirs.user_data_dir("polyglot", appauthor=False))


def log_dir() -> Path:
    import platformdirs

    return Path(platformdirs.user_log_dir("polyglot", appauthor=False))


def default_config_path() -> Path:
    return config_dir() / "config.toml"


# ----------------------------------------------------------------- loading
def _env_overrides(environ: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX) or "__" not in key:
            continue
        section, _, field = key[len(ENV_PREFIX) :].lower().partition("__")
        if section not in SECTIONS:
            continue
        value: Any = raw
        try:  # numbers, booleans, null as JSON; anything else stays a string
            value = json.loads(raw)
        except json.JSONDecodeError:
            pass
        if isinstance(value, (dict, list)):
            value = raw
        out.setdefault(section, {})[field] = value
    return out


def _merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None = None, environ: Mapping[str, str] | None = None) -> AppConfig:
    path = path or default_config_path()
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"cannot read {path}: {exc}") from exc
    data = _merge(data, _env_overrides(os.environ if environ is None else environ))
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration ({path}): {exc}") from exc


# ----------------------------------------------------------------- saving
def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # a JSON string is a valid TOML basic string
    raise TypeError(f"cannot write {type(value).__name__} to TOML")


def to_toml(config: AppConfig) -> str:
    lines = ["# Polyglot configuration. API keys are stored in the OS keyring, not here.", ""]
    for section, values in config.model_dump(mode="json").items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            if value is None:
                continue  # TOML has no null; absence means "default"
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def save_config(config: AppConfig, path: Path | None = None) -> Path:
    """Write the whole config once, atomically (temp file + rename)."""
    path = path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".config-", suffix=".toml", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(to_toml(config))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def changed_sections(old: AppConfig, new: AppConfig) -> set[str]:
    return {s for s in SECTIONS if getattr(old, s) != getattr(new, s)}


# ----------------------------------------------------------------- secrets
class Secrets:
    OPENAI = "openai_api_key"
    GOOGLE = "google_api_key"
    _ENV: Mapping[str, tuple[str, ...]] = {
        OPENAI: ("POLYGLOT_OPENAI_API_KEY", "OPENAI_API_KEY"),
        GOOGLE: ("POLYGLOT_GOOGLE_API_KEY", "GOOGLE_TRANSLATE_API_KEY"),
    }

    def __init__(self, store: SecretStore, environ: Mapping[str, str] | None = None) -> None:
        self._store = store
        self._environ = os.environ if environ is None else environ

    def get(self, name: str) -> str | None:
        for var in self._ENV.get(name, ()):
            value = self._environ.get(var)
            if value:
                return value
        return self._store.get(name)

    def set(self, name: str, value: str | None) -> None:
        self._store.set(name, value)
