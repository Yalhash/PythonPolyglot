"""Logging: one logger per module, a rotating file in the user log directory, stderr.

Transcript text is never logged by default. INFO and above carry IDs, timings and sizes.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def configure(level: str = "INFO", log_file: Path | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(threadName)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.handlers[:] = [stream]
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
        except OSError as exc:
            root.warning("cannot write log file %s: %s", log_file, exc)
        else:
            fh.setFormatter(fmt)
            root.addHandler(fh)
    for noisy in ("faster_whisper", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
