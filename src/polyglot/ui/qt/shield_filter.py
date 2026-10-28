"""Apply the capture shield to every top-level window the app shows.

In Qt, dialogs, menus, combo-box popups and tooltips are separate top-level windows,
each with its own native handle. Shielding only the main window would expose a settings
dialog opened during a screen share. This event filter is installed on the
QApplication and shields each window when it is shown, and again whenever Qt replaces
its native handle (``WinIdChange``).

``winId()`` is only called once the native window exists: called earlier it *creates*
one, and on a window that is half created or already being torn down (a tooltip during a
click) PyQt returns ``None``. A window without a handle yet is skipped; Qt sends
``WinIdChange`` once the handle exists.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget

from polyglot.core.ports import ShieldStatus, WindowShield

log = logging.getLogger(__name__)

_ORDER = {ShieldStatus.HIDDEN: 0, ShieldStatus.UNSUPPORTED: 1, ShieldStatus.FAILED: 2}


def native_handle(widget: QWidget) -> int | None:
    """The widget's existing native window handle, or None. Never creates one.

    ``winId()`` only returns the existing handle once Qt has created the native window
    (``WA_WState_Created`` and a ``windowHandle()``); before that it would create one.
    """
    if sip.isdeleted(widget):
        return None
    if not widget.testAttribute(Qt.WidgetAttribute.WA_WState_Created):
        return None
    if widget.windowHandle() is None:  # no native window (yet, or any more)
        return None
    handle = widget.winId()
    if handle is None:  # PyQt returns None for a null handle
        return None
    value = int(handle)
    return value or None


class CaptureShieldFilter(QObject):
    #: The worst status across all shown windows (FAILED > UNSUPPORTED > HIDDEN).
    status_changed = pyqtSignal(object)

    def __init__(self, shield: WindowShield) -> None:
        super().__init__()
        self._shield = shield
        self._applied: dict[int, tuple[int, ShieldStatus]] = {}  # id(widget) -> (hwnd, status)
        self._watched: list[tuple[QWidget, Callable[..., None]]] = []
        self._failed_once: set[int] = set()
        self._closed = False
        self.status: ShieldStatus | None = None

    def shield_widget(self, widget: QWidget) -> ShieldStatus | None:
        """Shield one top-level window. Returns None if it has no native handle yet."""
        if self._closed:
            return None
        win_id = native_handle(widget)
        if win_id is None:
            log.debug("no native handle yet for %s; waiting for WinIdChange", _describe(widget))
            return None
        key = id(widget)
        known = self._applied.get(key)
        if known is not None and known[0] == win_id:
            return known[1]
        status = self._shield.exclude_from_capture(win_id)
        if known is None:
            slot = self._forgetter(key)
            widget.destroyed.connect(slot)
            self._watched.append((widget, slot))
        self._applied[key] = (win_id, status)
        if status is not ShieldStatus.HIDDEN:
            log.info("window %s not hidden from capture: %s", _describe(widget), status.value)
        self._update()
        return status

    def _forgetter(self, key: int) -> Callable[..., None]:
        def forget(*_: object) -> None:
            if self._closed:
                return
            self._watched = [(w, s) for w, s in self._watched if id(w) != key]
            if self._applied.pop(key, None) is not None:
                self._update()

        return forget

    def _update(self) -> None:
        statuses = [s for _, s in self._applied.values()]
        worst = max(statuses, key=_ORDER.__getitem__, default=None)
        if worst != self.status:
            self.status = worst
            self.status_changed.emit(worst)

    def close(self) -> None:
        """Stop for good: no more shielding and no more ``status_changed``. Call before
        the windows are deleted, so their destruction cannot reach a deleted view."""
        if self._closed:
            return
        self._closed = True
        for widget, slot in self._watched:
            if not sip.isdeleted(widget):
                with contextlib.suppress(TypeError):  # already disconnected
                    widget.destroyed.disconnect(slot)
        self._watched.clear()
        self._applied.clear()
        with contextlib.suppress(TypeError):  # nothing connected
            self.status_changed.disconnect()

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        if (
            not self._closed
            and event is not None
            and event.type() in (QEvent.Type.Show, QEvent.Type.WinIdChange)
            and isinstance(obj, QWidget)
            and not sip.isdeleted(obj)
            and obj.isWindow()
        ):
            try:
                self.shield_widget(obj)
            except Exception as exc:
                key = id(obj)
                if key not in self._failed_once:  # one line per window, not one per click
                    self._failed_once.add(key)
                    log.warning("capture shield failed for %s: %r", _describe(obj), exc)
        return False


def _describe(widget: QWidget) -> str:
    if sip.isdeleted(widget):
        return "<deleted widget>"
    name = widget.objectName()
    return f"{type(widget).__name__}({name!r})" if name else type(widget).__name__
