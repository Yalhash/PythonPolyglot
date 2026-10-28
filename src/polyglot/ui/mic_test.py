"""The microphone test: record a few seconds, judge the level, play it back.

Qt-free so it is unit-tested; the settings dialog is a thin shell around it. It runs the
(blocking) recording and playback on a worker thread and reports progress through a
callback, which the dialog turns into a queued Qt signal.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from polyglot.core.errors import PolyglotError
from polyglot.core.models import AudioClip

log = logging.getLogger(__name__)

TEST_SECONDS = 5.0

LISTENING_MESSAGE = (
    "Stop listening first: the playback would be picked up as the other person's audio."
)

_LEVEL_TEXT = {
    "ok": "Sounds good.",
    "quiet": "Very quiet. Raise the microphone volume, or move closer.",
    "clipping": "Too loud, it is distorting. Lower the microphone volume.",
}
_SILENT_TEXT = "No sound picked up. Check that the right microphone is selected and not muted."


class MicTester(Protocol):
    """Where recording and playback come from (built by the composition root, so the UI
    never imports an adapter)."""

    def record(self, device: int | str | None, seconds: float) -> AudioClip:
        """Block while recording from ``device`` (None = the default input).
        Raises ``PolyglotError`` on failure."""

    def play(self, clip: AudioClip) -> None:
        """Block while playing ``clip`` on the default output. Raises ``PolyglotError``."""


@dataclass(frozen=True)
class MicTestUpdate:
    text: str
    level: str = "info"  # "info" | "warn" | "error"
    busy: bool = False  # a test is still running


def _device(value: int | str | None) -> int | str | None:
    return int(value) if isinstance(value, str) and value.isdigit() else value


class MicTestRunner:
    def __init__(
        self,
        tester: MicTester,
        notify: Callable[[MicTestUpdate], None],
        seconds: float = TEST_SECONDS,
    ) -> None:
        self._tester, self._notify, self._seconds = tester, notify, seconds
        self._lock = threading.Lock()
        self._running = False
        self._cancelled = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self, device: int | str | None) -> bool:
        """Begin a test. False if one is already running."""
        with self._lock:
            if self._running:
                return False
            self._running, self._cancelled = True, False
        threading.Thread(
            target=self._run, args=(_device(device),), name="polyglot-mic-test", daemon=True
        ).start()
        return True

    def cancel(self) -> None:
        """Stop reporting and skip playback (the dialog was closed)."""
        self._cancelled = True

    # ------------------------------------------------------------------ worker
    def _say(self, text: str, level: str = "info", *, busy: bool = False) -> None:
        if not self._cancelled:
            self._notify(MicTestUpdate(text, level, busy))

    def _run(self, device: int | str | None) -> None:
        try:
            self._say(f"Recording for {self._seconds:g} s. Say something…", busy=True)
            clip = self._tester.record(device, self._seconds)
            level = clip.level()
            if level == "silent":
                self._say(_SILENT_TEXT, "warn")
            elif self._cancelled:
                pass
            else:
                self._say("Playing it back…", busy=True)
                self._tester.play(clip)
                self._say(_LEVEL_TEXT[level], "info" if level == "ok" else "warn")
        except PolyglotError as exc:
            log.info("microphone test failed: %s", exc)
            self._say(f"Microphone test failed: {exc}", "error")
        except Exception as exc:  # a driver error must not leave the button disabled
            log.exception("microphone test crashed")
            self._say(f"Microphone test failed: {exc}", "error")
        finally:
            self._running = False
