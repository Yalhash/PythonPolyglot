"""Apply the capture shield to every top-level window the app shows.

In Qt, dialogs, menus, combo-box popups and tooltips are separate top-level windows,
each with its own native handle. Shielding only the main window would expose a settings
dialog opened during a screen share. This event filter is installed on the
QApplication and shields each window when it is shown, and again whenever Qt replaces
its native handle (``WinIdChange``).

Three rules keep it safe:

* ``winId()`` is only called once the native window exists. Called earlier it *creates*
  one, and on a window that is half created or being torn down PyQt returns ``None``.
* ``WinIdChange`` is also sent while a window is being destroyed, when the object must
  not be touched. So it is never handled on the spot: it schedules a re-scan of the
  application's live top-level windows on the next event-loop turn.
* No widget references are kept. Tooltips and menus are created and deleted by Qt in
  C++, where PyQt cannot always tell that a wrapper has gone stale; only the ``destroyed``
  signal (which carries no object) is used to forget a window.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QWidget

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
        self._failed_once: set[int] = set()
        self._rescan_pending = False
        self._closed = False
        self.status: ShieldStatus | None = None

    # ------------------------------------------------------------ shielding
    def shield_widget(self, widget: QWidget) -> ShieldStatus | None:
        """Shield one live top-level window. Returns None if it has no native handle yet."""
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
            widget.destroyed.connect(self._forgetter(key))
        self._applied[key] = (win_id, status)
        if status is not ShieldStatus.HIDDEN:
            log.info("window %s not hidden from capture: %s", _describe(widget), status.value)
        self._update()
        return status

    def rescan(self) -> None:
        """Shield every visible top-level window, using fresh objects from Qt."""
        self._rescan_pending = False
        if self._closed:
            return
        for widget in QApplication.topLevelWidgets():
            if widget.isVisible():
                self._shield_safely(widget)

    def _shield_safely(self, widget: QWidget) -> None:
        try:
            self.shield_widget(widget)
        except Exception as exc:
            key = id(widget)
            if key not in self._failed_once:  # one line per window, not one per event
                self._failed_once.add(key)
                log.warning("capture shield failed for %s: %r", _describe(widget), exc)

    def _schedule_rescan(self) -> None:
        if not self._rescan_pending and not self._closed:
            self._rescan_pending = True
            QTimer.singleShot(0, self.rescan)

    # ------------------------------------------------------------ bookkeeping
    def _forgetter(self, key: int) -> Callable[..., None]:
        def forget(*_: object) -> None:
            if self._closed:
                return
            self._failed_once.discard(key)
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
        self._closed = True  # pending forgetters and re-scans become no-ops
        self._applied.clear()
        with contextlib.suppress(TypeError):  # nothing connected
            self.status_changed.disconnect()

    # ------------------------------------------------------------ event filter
    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        if self._closed or event is None:
            return False
        kind = event.type()
        if kind == QEvent.Type.WinIdChange:
            # May arrive from a window's destructor: never touch ``obj`` here.
            self._schedule_rescan()
        elif kind == QEvent.Type.Show and isinstance(obj, QWidget) and obj.isWindow():
            self._shield_safely(obj)  # a window being shown is alive
        return False


def _describe(widget: QWidget) -> str:
    if sip.isdeleted(widget):
        return "<deleted widget>"
    name = widget.objectName()
    return f"{type(widget).__name__}({name!r})" if name else type(widget).__name__
