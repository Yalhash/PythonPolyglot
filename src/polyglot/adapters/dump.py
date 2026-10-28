"""Optional debugging aid: save each final utterance's audio as a WAV file.

Off by default (``privacy.dump_audio``). Files go to the user data directory, never the
repository, and are named by utterance ID so they line up with the logs.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from polyglot.adapters.audio.playback import write_wav
from polyglot.core.models import SAMPLE_RATE, SpeechSegment

log = logging.getLogger(__name__)


class SegmentDumper:
    def __init__(self, folder: Path) -> None:
        self.folder = folder / time.strftime("%Y%m%d-%H%M%S")
        self.folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, segment: SpeechSegment) -> None:
        path = self.folder / f"utt-{segment.utterance_id:05d}.wav"
        write_wav(path, segment.audio, SAMPLE_RATE)
        log.debug("dumped %s", path)
