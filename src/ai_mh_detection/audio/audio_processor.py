from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(frozen=True)
class AudioFeatures:
    sr: int
    duration_s: float
    rms_mean: float
    zcr_mean: float


@dataclass(frozen=True)
class AudioProcessor:
    target_sr: int = 16000
    max_seconds: int = 30

    def load(self, path: str | Path) -> tuple[np.ndarray, int]:
        y, sr = librosa.load(str(path), sr=self.target_sr, mono=True, duration=self.max_seconds)
        if y is None or len(y) == 0:
            return np.zeros(self.target_sr, dtype=np.float32), self.target_sr
        return y.astype(np.float32), sr

    def featurize(self, y: np.ndarray, sr: int) -> AudioFeatures:
        duration_s = float(len(y) / max(sr, 1))
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        return AudioFeatures(
            sr=sr,
            duration_s=duration_s,
            rms_mean=float(rms),
            zcr_mean=float(zcr),
        )
