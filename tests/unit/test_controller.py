from __future__ import annotations

from typing import Any

from polyglot.adapters.secrets import MemorySecretStore
from polyglot.config import AppConfig, Secrets
from polyglot.controller import AppController
from polyglot.core.errors import ConfigError
from polyglot.core.events import EventBus, StageError
from polyglot.core.models import IdSource
from polyglot.core.session import Session
from polyglot.wiring import Built


class FakePipeline:
    def __init__(self, session: Session | None, ids: IdSource | None) -> None:
        self.session = session or Session("en")
        self.ids = ids or IdSource()
        self.running = False
        self.calls: list[str] = []

    def start(self) -> None:
        self.running = True
        self.calls.append("start")

    def stop(self) -> None:
        self.running = False
        self.calls.append("stop")

    def close(self) -> None:
        self.running = False
        self.calls.append("close")

    def __getattr__(self, name: str) -> Any:
        return lambda *a: self.calls.append(name)


class FakeBuilder:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.built: list[FakePipeline] = []

    def __call__(self, cfg: AppConfig, secrets: Secrets, bus: EventBus, **kw: Any) -> Built:
        if self.fail:
            raise ConfigError("no audio device")
        p = FakePipeline(kw.get("session"), kw.get("ids"))
        self.built.append(p)
        return Built(p, ["a warning"], translation_enabled=True)  # type: ignore[arg-type]


def make(builder: FakeBuilder | None = None) -> tuple[AppController, FakeBuilder, list[AppConfig], list[Any]]:
    builder = builder or FakeBuilder()
    saved: list[AppConfig] = []
    bus = EventBus()
    events: list[Any] = []
    bus.subscribe(events.append)
    c = AppController(AppConfig(), Secrets(MemorySecretStore(), environ={}), bus, builder=builder, save=saved.append)
    return c, builder, saved, events


def test_build_is_lazy_and_once() -> None:
    c, b, _, events = make()
    assert b.built == []
    c.start()
    c.start()
    assert len(b.built) == 1 and c.listening
    assert any(isinstance(e, StageError) and e.message == "a warning" for e in events)
    c.toggle()
    assert not c.listening


def test_commands_forwarded() -> None:
    c, b, _, _ = make()
    c.request_suggestion()
    c.set_language("es")
    c.clear()
    assert b.built[0].calls == ["request_suggestion", "set_language", "clear"]


def test_display_only_change_saves_without_rebuild() -> None:
    c, b, saved, _ = make()
    c.start()
    new = c.config.model_copy(deep=True)
    new.ui.opacity = 0.5
    assert c.apply_config(new) == {"ui"}
    assert len(b.built) == 1 and saved == [new]
    assert c.apply_config(new) == set()  # no change, no save
    assert len(saved) == 1


def test_pipeline_change_rebuilds_keeping_conversation_and_listening() -> None:
    c, b, _, _ = make()
    c.start()
    old = b.built[0]
    new = c.config.model_copy(deep=True)
    new.asr.model = "small"
    assert c.apply_config(new) == {"asr"}
    assert len(b.built) == 2
    fresh = b.built[1]
    assert old.calls[-1] == "close"
    assert fresh.session is old.session and fresh.ids is old.ids  # conversation continues
    assert fresh.running  # was listening -> listening again


def test_rebuild_when_not_listening_stays_stopped() -> None:
    c, b, _, _ = make()
    c.ensure_built()
    c.rebuild()
    assert len(b.built) == 2 and not b.built[1].running


def test_build_failure_is_reported_not_raised() -> None:
    c, _, _, events = make(FakeBuilder(fail=True))
    c.start()
    assert not c.listening
    assert any(isinstance(e, StageError) and "no audio device" in e.message for e in events)


def test_shutdown_closes() -> None:
    c, b, _, _ = make()
    c.start()
    c.shutdown()
    assert b.built[0].calls[-1] == "close" and c.built is None
