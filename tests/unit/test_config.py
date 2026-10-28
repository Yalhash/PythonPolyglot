from __future__ import annotations

import sys
from pathlib import Path

import pytest

from polyglot.adapters.secrets import MemorySecretStore
from polyglot.config import (
    AppConfig,
    Secrets,
    changed_sections,
    load_config,
    save_config,
    to_toml,
)
from polyglot.core.errors import ConfigError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


def test_defaults_are_private_by_default(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "missing.toml", environ={})
    assert cfg == AppConfig()
    assert not cfg.translation.enabled and not cfg.suggestions.enabled  # cloud is opt-in
    assert not cfg.privacy.dump_audio and not cfg.privacy.log_content


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    cfg = AppConfig()
    cfg.asr.model = "small"
    cfg.translation.target_language = "fr"
    cfg.ui.hotkey_suggest = 'ctrl+"quoted"'
    cfg.audio.file = "C:\\Users\\me\\a b.wav"
    save_config(cfg, path)
    assert load_config(path, environ={}) == cfg
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]  # no temp files left


def test_toml_output_is_valid_and_skips_none() -> None:
    text = to_toml(AppConfig())
    data = tomllib.loads(text)
    assert set(data) == {"audio", "vad", "asr", "translation", "suggestions", "ui", "privacy"}
    assert "language" not in data["asr"]  # None -> omitted
    assert "api_key" not in text


def test_precedence_env_over_file_over_default(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[asr]\nmodel = "small"\nbeam_size = 3\n[ui]\nopacity = 0.5\n', encoding="utf-8"
    )
    env = {
        "POLYGLOT_ASR__MODEL": "medium",
        "POLYGLOT_TRANSLATION__ENABLED": "true",
        "POLYGLOT_VAD__END_SILENCE_MS": "800",
        "POLYGLOT_NOT_A_SECTION__X": "1",
        "UNRELATED": "1",
    }
    cfg = load_config(path, environ=env)
    assert cfg.asr.model == "medium"  # env beats file
    assert cfg.asr.beam_size == 3  # file beats default
    assert cfg.ui.opacity == 0.5
    assert cfg.translation.enabled is True
    assert cfg.vad.end_silence_ms == 800


@pytest.mark.parametrize(
    "content",
    [
        "[asr]\nbeam_size = 99\n",
        "[vad]\nstart_threshold = 0.3\nend_threshold = 0.6\n",
        "[asr]\nunknown = 1\n",
        "not toml [",
    ],
)
def test_invalid_config_raises_config_error(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path, environ={})


def test_default_sections_are_independent_per_instance() -> None:
    a, b = AppConfig(), AppConfig()
    a.audio.ring_seconds = 99
    assert b.audio.ring_seconds == 30.0


def test_changed_sections() -> None:
    a = AppConfig()
    b = a.model_copy(deep=True)
    assert changed_sections(a, b) == set()
    b.ui.opacity = 0.5
    b.asr.model = "tiny"
    assert changed_sections(a, b) == {"ui", "asr"}


def test_secrets_env_overrides_store() -> None:
    store = MemorySecretStore({Secrets.OPENAI: "from-store"})
    assert Secrets(store, environ={}).get(Secrets.OPENAI) == "from-store"
    assert Secrets(store, environ={"OPENAI_API_KEY": "env"}).get(Secrets.OPENAI) == "env"
    assert (
        Secrets(store, environ={"POLYGLOT_OPENAI_API_KEY": "p", "OPENAI_API_KEY": "env"}).get(
            Secrets.OPENAI
        )
        == "p"
    )
    s = Secrets(store, environ={})
    s.set(Secrets.GOOGLE, "g")
    assert s.get(Secrets.GOOGLE) == "g"
    s.set(Secrets.GOOGLE, None)
    assert s.get(Secrets.GOOGLE) is None


def test_store_keeps_env_overrides_out_of_the_file(tmp_path: Path) -> None:
    from polyglot.config import ConfigStore

    path = tmp_path / "config.toml"
    path.write_text('[asr]\nmodel = "small"\n', encoding="utf-8")
    store = ConfigStore(path, environ={"POLYGLOT_TRANSLATION__ENABLED": "true", "POLYGLOT_ASR__MODEL": "tiny"})
    cfg = store.load()
    assert cfg.translation.enabled and cfg.asr.model == "tiny"
    cfg.ui.opacity = 0.5  # a real change from the settings dialog
    store.save(cfg)
    saved = load_config(path, environ={})
    assert saved.ui.opacity == 0.5
    assert saved.translation.enabled is False  # env value not persisted
    assert saved.asr.model == "small"  # file value kept under the env override


def test_store_saves_a_dialog_change_to_an_env_overridden_field(tmp_path: Path) -> None:
    from polyglot.config import ConfigStore

    path = tmp_path / "config.toml"
    store = ConfigStore(path, environ={"POLYGLOT_ASR__MODEL": "tiny"})
    cfg = store.load()
    cfg.asr.model = "medium"
    store.save(cfg)
    assert load_config(path, environ={}).asr.model == "medium"


def test_store_refuses_to_overwrite_an_unloadable_file(tmp_path: Path) -> None:
    from polyglot.config import ConfigStore

    path = tmp_path / "config.toml"
    path.write_text("[asr]\nbeam_size = 99\n", encoding="utf-8")
    store = ConfigStore(path, environ={})
    with pytest.raises(ConfigError):
        store.load()
    with pytest.raises(ConfigError):
        store.save(AppConfig())
    assert "beam_size = 99" in path.read_text(encoding="utf-8")


def test_numeric_device_and_text_env_values(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[audio]\ndevice = 3\n", encoding="utf-8")
    assert load_config(path, environ={}).audio.device == "3"
    cfg = load_config(tmp_path / "none.toml", environ={"POLYGLOT_AUDIO__DEVICE": "3", "POLYGLOT_UI__OPACITY": "0.5"})
    assert cfg.audio.device == "3" and cfg.ui.opacity == 0.5


def test_toml_escapes_delete_character() -> None:
    cfg = AppConfig()
    cfg.ui.hotkey_suggest = "a\x7fb"
    assert tomllib.loads(to_toml(cfg))["ui"]["hotkey_suggest"] == "a\x7fb"


def test_secret_source() -> None:
    store = MemorySecretStore({Secrets.GOOGLE: "g"})
    s = Secrets(store, environ={"OPENAI_API_KEY": "e"})
    assert s.source(Secrets.OPENAI) == "OPENAI_API_KEY"
    assert s.source(Secrets.GOOGLE) == "keychain"
    assert Secrets(MemorySecretStore(), environ={}).source(Secrets.OPENAI) is None
