"""Application controller: owns the pipeline's lifecycle and applies config changes.

Qt-free, so it is unit-tested. The UI calls it through the ``Commands`` protocol the
presenter uses; it never touches widgets.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from polyglot.config import AppConfig, Secrets, changed_sections, save_config
from polyglot.core.errors import ConfigError, PolyglotError
from polyglot.core.events import EventBus, PipelineState, PipelineStatus, StageError
from polyglot.core.models import ErrorKind
from polyglot.wiring import Built, build, build_demo

log = logging.getLogger(__name__)

#: Sections whose change requires rebuilding the pipeline (and stopping capture).
PIPELINE_SECTIONS = frozenset({"audio", "vad", "asr", "translation", "suggestions"})

Builder = Callable[..., Built]


class AppController:
    def __init__(
        self,
        config: AppConfig,
        secrets: Secrets,
        bus: EventBus,
        *,
        demo: bool = False,
        builder: Builder | None = None,
        save: Callable[[AppConfig], object] | None = save_config,
    ) -> None:
        self.config, self.secrets, self.bus, self.demo = config, secrets, bus, demo
        self._builder: Builder = builder or (
            (lambda cfg, _secrets, bus, **kw: build_demo(cfg, bus, **kw)) if demo else build
        )
        self._save = save
        self.built: Built | None = None
        self.warnings: list[str] = []

    # ---------------------------------------------------------------- building
    def ensure_built(self) -> Built | None:
        if self.built is not None:
            return self.built
        try:
            self.built = self._builder(self.config, self.secrets, self.bus)
        except (ConfigError, PolyglotError) as exc:
            log.warning("cannot build pipeline: %s", exc)
            self.bus.publish(StageError("setup", exc.kind, str(exc)))
            self.bus.publish(PipelineStatus(PipelineState.ERROR, str(exc)))
            return None
        except Exception as exc:
            log.exception("unexpected error building the pipeline")
            self.bus.publish(StageError("setup", ErrorKind.INTERNAL, str(exc)))
            self.bus.publish(PipelineStatus(PipelineState.ERROR, str(exc)))
            return None
        self.warnings = list(self.built.warnings)
        for w in self.warnings:
            self.bus.publish(StageError("setup", ErrorKind.NOT_CONFIGURED, w))
        return self.built

    def _carry(self, previous: Built | None) -> dict[str, object]:
        if previous is None:
            return {}
        return {"session": previous.pipeline.session, "ids": previous.pipeline.ids}

    # ---------------------------------------------------------------- Commands
    @property
    def listening(self) -> bool:
        return self.built is not None and self.built.pipeline.running

    def start(self) -> None:
        built = self.ensure_built()
        if built is not None:
            built.pipeline.start()

    def stop(self) -> None:
        if self.built is not None:
            self.built.pipeline.stop()

    def toggle(self) -> None:
        if self.listening:
            self.stop()
        else:
            self.start()

    def request_suggestion(self) -> None:
        built = self.ensure_built()
        if built is not None:
            built.pipeline.request_suggestion()

    def set_language(self, language: str | None) -> None:
        built = self.ensure_built()
        if built is not None:
            built.pipeline.set_language(language)

    def clear(self) -> None:
        if self.built is not None:
            self.built.pipeline.clear()

    def shutdown(self) -> None:
        if self.built is not None:
            self.built.pipeline.close()
            self.built = None

    # ---------------------------------------------------------------- settings
    def apply_config(self, new: AppConfig) -> set[str]:
        """Apply and persist a new config. Returns the sections that changed.

        Display-only sections apply live (the UI reads them). Pipeline sections rebuild
        the pipeline, keeping the conversation; capture restarts if it was running.
        """
        changed = changed_sections(self.config, new)
        if not changed:
            return changed
        self.config = new
        if self._save is not None:
            try:
                self._save(new)
            except OSError as exc:
                self.bus.publish(StageError("settings", ErrorKind.INTERNAL, f"could not save settings: {exc}"))
        if changed & PIPELINE_SECTIONS:
            self.rebuild()
        return changed

    def rebuild(self) -> None:
        """Rebuild the pipeline from the current config and secrets, keeping the
        conversation. Restarts listening if it was on. Used after settings or API keys
        change."""
        if self.built is None:
            return
        was_listening = self.listening
        previous = self.built
        previous.pipeline.close()
        self.built = None
        try:
            self.built = self._builder(self.config, self.secrets, self.bus, **self._carry(previous))
        except (ConfigError, PolyglotError) as exc:
            self.bus.publish(StageError("setup", exc.kind, str(exc)))
            self.bus.publish(PipelineStatus(PipelineState.ERROR, str(exc)))
            return
        self.warnings = list(self.built.warnings)
        for w in self.warnings:
            self.bus.publish(StageError("setup", ErrorKind.NOT_CONFIGURED, w))
        if was_listening:
            self.built.pipeline.start()
