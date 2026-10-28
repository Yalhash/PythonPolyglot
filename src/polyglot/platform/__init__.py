"""OS-specific services, selected once. No other module checks the platform."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from polyglot.core.ports import ShieldStatus, WindowShield


class UnsupportedShield:
    """macOS 15+ (ScreenCaptureKit ignores NSWindow.sharingType) and Linux: by decision
    the overlay is not hidden there; the UI shows a "visible in screen share" indicator."""

    def exclude_from_capture(self, window_id: int) -> ShieldStatus:
        return ShieldStatus.UNSUPPORTED


@dataclass(frozen=True)
class PlatformInfo:
    name: str  # "windows" | "macos" | "linux"
    has_loopback: bool  # can capture system audio without a virtual device


def current() -> PlatformInfo:
    if sys.platform == "win32":
        return PlatformInfo("windows", has_loopback=True)
    if sys.platform == "darwin":
        return PlatformInfo("macos", has_loopback=False)
    return PlatformInfo("linux", has_loopback=False)


def window_shield() -> WindowShield:
    if sys.platform == "win32":
        from polyglot.platform.windows import WindowsShield

        return WindowsShield()
    return UnsupportedShield()
