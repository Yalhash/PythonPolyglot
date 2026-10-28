"""Show a real window and report what the capture shield did (CI notice annotation).

Usage: python scripts/shield_check.py   (Windows; native Qt platform)
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QWidget

from polyglot.core.ports import ShieldStatus
from polyglot.platform import window_shield


def main() -> int:
    app = QApplication(sys.argv)
    w = QWidget()
    w.resize(200, 100)
    w.show()
    app.processEvents()
    shield = window_shield()
    status = shield.exclude_from_capture(int(w.winId()))
    detail = getattr(shield, "detail", "")
    build = getattr(shield, "build", "?")
    level = "notice" if status is ShieldStatus.HIDDEN else "error"
    print(
        f"::{level} title=Capture shield::{status.value} on {app.platformName()} build {build} {detail}"
    )
    w.close()
    return 0 if status is ShieldStatus.HIDDEN else 1


if __name__ == "__main__":
    sys.exit(main())
