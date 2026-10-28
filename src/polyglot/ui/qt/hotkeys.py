"""System-wide hotkeys (work while another app, e.g. the call, has focus).

Uses pynput. If it cannot start (no permission on macOS, no display), the app keeps
working with the in-window buttons and logs why.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

log = logging.getLogger(__name__)


class GlobalHotkeys(QObject):
    suggest = pyqtSignal()
    toggle_window = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._listener: Any = None

    def bind(self, suggest_combo: str, toggle_combo: str) -> bool:
        self.stop()
        if not suggest_combo and not toggle_combo:
            return False
        try:
            from pynput import keyboard
        except Exception as exc:  # ImportError, or a backend error on import
            log.warning("global hotkeys unavailable: %s", exc)
            return False
        mapping = {}
        if suggest_combo:
            mapping[suggest_combo] = self.suggest.emit  # emitted from pynput's thread: queued
        if toggle_combo:
            mapping[toggle_combo] = self.toggle_window.emit
        try:
            self._listener = keyboard.GlobalHotKeys(mapping)
            self._listener.start()
        except Exception as exc:
            log.warning("could not register global hotkeys %s: %s", list(mapping), exc)
            self._listener = None
            return False
        return True

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
