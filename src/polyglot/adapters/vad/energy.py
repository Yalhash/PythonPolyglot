"""Energy-based voice activity: no model, no dependencies.

Tracks the background noise floor and maps loudness above it to a probability. Much less
accurate than Silero in noisy rooms or with music, but useful as a fallback when the
Silero model cannot be loaded, and as a deterministic VAD in tests.
"""

from __future__ import annotations

import math

import numpy as np

from polyglot.core.ports import RawAudio


class EnergyVad:
    def __init__(
        self,
        margin_db: float = 12.0,
        slope_db: float = 3.0,
        initial_floor_db: float = -60.0,
        adapt: float = 0.05,
    ) -> None:
        self.margin_db, self.slope_db = margin_db, slope_db
        self.initial_floor_db, self.adapt = initial_floor_db, adapt
        self.reset()

    def reset(self) -> None:
        self.floor_db = self.initial_floor_db

    def speech_prob(self, frame: RawAudio) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if frame.size else 0.0
        db = 20.0 * math.log10(max(rms, 1e-9))
        above = db - (self.floor_db + self.margin_db)
        if db < self.floor_db:
            self.floor_db = max(db, -100.0)  # follow quiet rooms down immediately
        elif above < 0:
            self.floor_db += self.adapt * (db - self.floor_db)  # drift up slowly with noise
        return 1.0 / (1.0 + math.exp(-above / self.slope_db))
