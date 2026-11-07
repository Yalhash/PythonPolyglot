"""Entry script for the packaged app (PyInstaller).

A windowed build has no console, so ``sys.stdout`` and ``sys.stderr`` are ``None``.
Anything that writes to them - logging's stderr handler, the progress bar that
huggingface_hub shows while downloading the speech model on first run, ``--version`` -
would then crash with ``AttributeError: 'NoneType' object has no attribute 'write'``.
Point them at the null device instead; the real log still goes to the log file.
"""

from __future__ import annotations

import os
import sys


def _ensure_streams() -> None:
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115


if __name__ == "__main__":
    _ensure_streams()

    from polyglot.__main__ import main

    sys.exit(main())
