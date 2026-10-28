"""Apply the capture shield to every top-level window the app shows.

In Qt, dialogs, menus, combo-box popups and tooltips are separate top-level windows,
each with its own native handle. Shielding only the main window would expose a settings
dialog opened during a screen share. This event filter is installed on the
QApplication and shields each window when it is shown, and again whenever Qt replaces
its native handle (``WinIdChange``).
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QEvent, QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget

from polyglot.core.ports import ShieldStatus, WindowShield

log = logging.getLogger(__name__)

_ORDER = {ShieldStatus.HIDDEN: 0, ShieldStatus.UNSUPPORTED: 1, ShieldStatus.FAILED: 2}


class CaptureShieldFilter(QObject):
    #: The worst status across all shown windows (FAILED > UNSUPPORTED > HIDDEN).
    status_changed = pyqtSignal(object)

    def __init__(self, shield: WindowShield) -> None:
        super().__init__()
        self._shield = shield
        self._applied: dict[int, tuple[int, ShieldStatus]] = {}  # id(widget) -> (hwnd, status)
        self.status: ShieldStatus | None = None

    def shield_widget(self, widget: QWidget) -> ShieldStatus:
        key, win_id = id(widget), int(widget.winId())
        known = self._applied.get(key)
        if known is not None and known[0] == win_id:
            return known[1]
        status = self._shield.exclude_from_capture(win_id)
        if known is None:
            widget.destroyed.connect(lambda *_: self._forget(key))
        self._applied[key] = (win_id, status)
        if status is not ShieldStatus.HIDDEN:
            log.info("window %r not hidden from capture: %s", widget.objectName(), status.value)
        self._update()
        return status

    def _forget(self, key: int) -> None:
        if self._applied.pop(key, None) is not None:
            self._update()

    def _update(self) -> None:
        statuses = [s for _, s in self._applied.values()]
        worst = max(statuses, key=_ORDER.__getitem__, default=None)
        if worst != self.status:
            self.status = worst
            self.status_changed.emit(worst)

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        if (
            event is not None
            and event.type() in (QEvent.Type.Show, QEvent.Type.WinIdChange)
            and isinstance(obj, QWidget)
            and obj.isWindow()
        ):
            try:
                self.shield_widget(obj)
            except Exception:
                log.exception("capture shield failed")
        return False
