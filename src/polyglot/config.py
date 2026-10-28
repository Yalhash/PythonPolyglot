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
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from polyglot.core.errors import ConfigError
from polyglot.core.ports import SecretStore
from polyglot.core.profile import UserProfile

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

    @field_validator("device", "user_device", mode="before")
    @classmethod
    def _device_as_text(cls, value: object) -> object:
        return str(value) if isinstance(value, int) and not isinstance(value, bool) else value

    file: str | None = None  # WAV path, for source = "file"
    #: Also listen to the user's own microphone, so suggestions know what they said.
    #: Opt-in. Audio stays on this computer; only transcribed text is ever sent anywhere.
    user_mic: bool = False
    user_device: str | None = None  # input device name or index; None = the default input
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
    #: auto = the fastest type the device supports (int8 on CPU, float16 on CUDA).
    #: "default" would keep the model's saved float16, which CPUs convert to slow float32.
    compute_type: Literal[
        "auto",
        "int8",
        "int8_float16",
        "int8_float32",
        "int8_bfloat16",
        "int16",
        "float16",
        "bfloat16",
        "float32",
    ] = "auto"
    beam_size: int = Field(5, ge=1, le=10)
    language: str | None = None  # force a source language; None = auto-detect then lock
    lock_after: int = Field(2, ge=1, le=10)
    lock_min_probability: float = Field(0.8, ge=0.0, le=1.0)
    queue_size: int = Field(8, ge=1, le=64)

    @field_validator("compute_type", mode="before")
    @classmethod
    def _legacy_default(cls, value: object) -> object:
        # Older versions saved "default" without the user choosing it; it caused a
        # float16 -> float32 conversion (and a CTranslate2 warning) on CPUs.
        return "auto" if value == "default" else value


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
    history: int = Field(6, ge=0, le=30)  # most utterances of earlier conversation
    history_chars: int = Field(3000, ge=200, le=20000)  # soft cap on their total length
    options: int = Field(2, ge=1, le=3)


_Short = Annotated[str, StringConstraints(strip_whitespace=True, max_length=120)]
_Long = Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)]
_Items = Annotated[list[_Short], Field(max_length=15)]


class ProfileConfig(_Section):
    """Public details the user lets the suggester use. Saved in ``config.toml`` like every
    other setting, so it is set up once. See ``polyglot.core.profile`` for what belongs
    here (and what deliberately has no field). Sent to the provider with each request."""

    name: _Short = ""
    role: _Short = ""
    location: _Short = ""
    summary: _Long = ""
    goals: _Short = ""
    tone: _Short = ""
    languages: _Items = Field(default_factory=list)
    skills: _Items = Field(default_factory=list)
    experience: _Items = Field(default_factory=list)
    education: _Items = Field(default_factory=list)
    hobbies: _Items = Field(default_factory=list)
    interests: _Items = Field(default_factory=list)
    talking_points: _Items = Field(default_factory=list)

    @field_validator(
        "languages",
        "skills",
        "experience",
        "education",
        "hobbies",
        "interests",
        "talking_points",
        mode="before",
    )
    @classmethod
    def _as_items(cls, value: object) -> object:
        # An environment variable arrives as text: one item per line or per ";".
        if isinstance(value, str):
            value = re.split(r"[\n;]", value)
        if isinstance(value, (list, tuple)):
            return [
                v.strip() if isinstance(v, str) else v
                for v in value
                if not (isinstance(v, str) and not v.strip())
            ]
        return value

    def to_core(self) -> UserProfile:
        return UserProfile(
            name=self.name,
            role=self.role,
            location=self.location,
            summary=self.summary,
            goals=self.goals,
            tone=self.tone,
            languages=tuple(self.languages),
            skills=tuple(self.skills),
            experience=tuple(self.experience),
            education=tuple(self.education),
            hobbies=tuple(self.hobbies),
            interests=tuple(self.interests),
            talking_points=tuple(self.talking_points),
        )


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
    audio: AudioConfig = AudioConfig()  # pydantic copies model defaults per instance
    vad: VadConfig = VadConfig()  # pydantic copies model defaults per instance
    asr: AsrConfig = AsrConfig()  # pydantic copies model defaults per instance
    translation: TranslationConfig = (
        TranslationConfig()
    )  # pydantic copies model defaults per instance
    suggestions: SuggestionsConfig = (
        SuggestionsConfig()
    )  # pydantic copies model defaults per instance
    profile: ProfileConfig = ProfileConfig()  # pydantic copies model defaults per instance
    ui: UiConfig = UiConfig()  # pydantic copies model defaults per instance
    privacy: PrivacyConfig = PrivacyConfig()  # pydantic copies model defaults per instance


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
        # Kept as text: pydantic converts "true"/"800"/"0.5" for typed fields, and text
        # fields (a device named "3") stay text.
        out.setdefault(section, {})[field] = raw
    return out


def _merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        # Typed explicitly: with mypy targeting 3.10, ``tomllib`` is the ``tomli`` fallback,
        # which is not installed on 3.11+ and so reads as Any.
        data: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    return data


def _validate(data: dict[str, Any], path: Path) -> AppConfig:
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration ({path}): {exc}") from exc


class ConfigStore:
    """Loads the layered config and saves only what belongs in the file.

    Environment overrides apply to the running app but are never written to
    ``config.toml``: on save, a field still equal to its env override keeps the file's
    own value. If the file could not be loaded, saving is refused, so a broken file is
    never silently replaced with defaults.
    """

    def __init__(self, path: Path | None = None, environ: Mapping[str, str] | None = None) -> None:
        self.path = path or default_config_path()
        self._environ = os.environ if environ is None else environ
        self._file: dict[str, Any] = {}
        self._env: dict[str, dict[str, Any]] = {}
        self._env_effective: dict[str, Any] = {}
        self.loaded = False

    def load(self) -> AppConfig:
        self.loaded = False
        file_data = _read_file(self.path)
        env = _env_overrides(self._environ)
        config = _validate(_merge(file_data, env), self.path)
        self._file, self._env = file_data, env
        self._env_effective = config.model_dump(mode="json")
        self.loaded = True
        return config

    def save(self, config: AppConfig) -> Path:
        if not self.loaded:
            raise ConfigError(f"not saving: {self.path} could not be loaded; fix or remove it")
        data = config.model_dump(mode="json")
        for section, fields in self._env.items():
            for field in fields:
                if field not in data.get(section, {}):
                    continue
                if data[section][field] == self._env_effective.get(section, {}).get(field):
                    data[section][field] = self._file.get(section, {}).get(field)  # file layer
        _write_atomic(self.path, dict_to_toml(data))
        self._file = {
            s: {k: v for k, v in vals.items() if v is not None} for s, vals in data.items()
        }
        return self.path


def load_config(path: Path | None = None, environ: Mapping[str, str] | None = None) -> AppConfig:
    return ConfigStore(path, environ).load()


# ----------------------------------------------------------------- saving
def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, str):
        # A JSON string is a TOML basic string, except that TOML also forbids a raw DEL.
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    raise TypeError(f"cannot write {type(value).__name__} to TOML")


def dict_to_toml(data: Mapping[str, Mapping[str, Any]]) -> str:
    lines = ["# Polyglot configuration. API keys are stored in the OS keyring, not here.", ""]
    for section, values in data.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            if value is None:
                continue  # TOML has no null; absence means "default"
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def to_toml(config: AppConfig) -> str:
    return dict_to_toml(config.model_dump(mode="json"))


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".config-", suffix=".toml", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_config(config: AppConfig, path: Path | None = None) -> Path:
    """Write the whole config once, atomically. (The app saves through ConfigStore,
    which also keeps environment overrides out of the file.)"""
    path = path or default_config_path()
    _write_atomic(path, to_toml(config))
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

    def source(self, name: str) -> str | None:
        """Where the effective value comes from: an env var name, "keychain", or None."""
        for var in self._ENV.get(name, ()):
            if self._environ.get(var):
                return var
        return "keychain" if self._store.get(name) else None

    def set(self, name: str, value: str | None) -> None:
        self._store.set(name, value)
