from __future__ import annotations

from typing import Any

import pytest

from polyglot.config import AppConfig
from polyglot.core.errors import AudioSourceError
from polyglot.wiring import build_user_source

SD = "polyglot.adapters.audio.sounddevice_source.SoundDeviceSource"


class FakeMic:
    made: list[Any] = []  # noqa: RUF012

    def __init__(self, device: Any = None) -> None:
        self.device = device
        self.device_name = f"mic:{device}"
        FakeMic.made.append(self)


@pytest.fixture(autouse=True)
def fake_mic(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeMic.made = []
    monkeypatch.setattr(SD, FakeMic)


def cfg(**audio: Any) -> AppConfig:
    c = AppConfig()
    for key, value in audio.items():
        setattr(c.audio, key, value)
    return c


def test_the_microphone_is_off_by_default() -> None:
    warnings: list[str] = []
    assert build_user_source(AppConfig(), warnings) is None
    assert warnings == [] and FakeMic.made == []


def test_no_microphone_alongside_a_replayed_file() -> None:
    warnings: list[str] = []
    assert build_user_source(cfg(user_mic=True, source="file", file="a.wav"), warnings) is None
    assert warnings == []


def test_builds_the_chosen_input_device() -> None:
    warnings: list[str] = []
    built = build_user_source(cfg(user_mic=True, source="loopback", user_device="3"), warnings)
    assert built is not None and built[1] == "mic:3"
    assert FakeMic.made[0].device == 3  # a numeric id is passed as a number
    assert warnings == []
    default = build_user_source(cfg(user_mic=True, source="loopback"), warnings)
    assert default is not None and FakeMic.made[1].device is None  # the default input


def test_refuses_to_use_the_main_input_as_the_microphone() -> None:
    warnings: list[str] = []
    c = cfg(user_mic=True, source="device", device="2", user_device="2")
    assert build_user_source(c, warnings) is None
    assert len(warnings) == 1 and "tagged as yours" in warnings[0]
    # A different device is fine.
    assert (
        build_user_source(cfg(user_mic=True, source="device", device="2", user_device="4"), [])
        is not None
    )


def test_an_unusable_microphone_is_a_warning_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(device: Any = None) -> None:
        raise AudioSourceError("input device 9 not found")

    monkeypatch.setattr(SD, broken)
    warnings: list[str] = []
    assert (
        build_user_source(cfg(user_mic=True, source="loopback", user_device="9"), warnings) is None
    )
    assert len(warnings) == 1 and "not found" in warnings[0]
