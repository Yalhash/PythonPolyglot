"""Windows: exclude a window from screen capture while it stays visible on the monitor.

Uses ``SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)``. Per Microsoft's docs:

* supported from Windows 10 version 2004 (build 19041); on older builds the same call
  behaves like ``WDA_MONITOR`` and shows a black box in captures, which would obstruct
  the viewer, so on old builds we do not call it at all;
* works on top-level windows only, and only while DWM composes the desktop.

The result is read back with ``GetWindowDisplayAffinity`` before reporting success.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from polyglot.core.ports import ShieldStatus

log = logging.getLogger(__name__)

WDA_NONE = 0x00
WDA_EXCLUDEFROMCAPTURE = 0x11
MIN_BUILD = 19041


def _load_user32() -> Any:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)  # type: ignore[attr-defined]
    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
    user32.GetWindowDisplayAffinity.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowDisplayAffinity.restype = wintypes.BOOL
    return user32


def _windows_build() -> int:
    getver = getattr(sys, "getwindowsversion", None)
    return int(getver().build) if getver else 0


class WindowsShield:
    def __init__(self, user32: Any = None, build: int | None = None, last_error: Any = None) -> None:
        self.build = _windows_build() if build is None else build
        self._user32 = user32
        self._last_error = last_error

    @property
    def supported(self) -> bool:
        return self.build >= MIN_BUILD

    def _api(self) -> Any:
        if self._user32 is None:
            self._user32 = _load_user32()
        return self._user32

    def _get_last_error(self) -> int:
        if self._last_error is not None:
            return int(self._last_error())
        import ctypes

        return int(ctypes.get_last_error())  # type: ignore[attr-defined]

    def exclude_from_capture(self, window_id: int) -> ShieldStatus:
        if not self.supported:
            log.warning("Windows build %d < %d: capture exclusion unsupported", self.build, MIN_BUILD)
            return ShieldStatus.UNSUPPORTED
        api = self._api()
        if not api.SetWindowDisplayAffinity(window_id, WDA_EXCLUDEFROMCAPTURE):
            log.warning("SetWindowDisplayAffinity failed for hwnd=%#x (error %d)", window_id, self._get_last_error())
            return ShieldStatus.FAILED
        import ctypes
        from ctypes import wintypes

        value = wintypes.DWORD(0)
        if not api.GetWindowDisplayAffinity(window_id, ctypes.byref(value)):
            log.warning("GetWindowDisplayAffinity failed for hwnd=%#x (error %d)", window_id, self._get_last_error())
            return ShieldStatus.FAILED
        if value.value != WDA_EXCLUDEFROMCAPTURE:
            log.warning("display affinity for hwnd=%#x reads back as %#x", window_id, value.value)
            return ShieldStatus.FAILED
        return ShieldStatus.HIDDEN
