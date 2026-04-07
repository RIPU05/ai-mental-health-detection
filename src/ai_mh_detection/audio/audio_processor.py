from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from pydub import AudioSegment


@dataclass(frozen=True)
class AudioFeatures:
    sr: int
    duration_s: float
    rms_mean: float
    zcr_mean: float


def load_audio(
    file_path: str | Path,
    *,
    target_sr: int = 16000,
    max_seconds: int = 30,
) -> tuple[np.ndarray, int]:
    """
    Load audio (WAV/MP3/etc) into a mono float32 waveform.

    MP3 handling:
    1) Try `librosa.load` first.
    2) If librosa fails to decode the MP3, convert MP3 -> temporary WAV using `pydub`.
    3) Load the temporary WAV with librosa.

    Temporary WAV cleanup:
    - The temporary WAV file is deleted in a `finally` block.
    - This prevents accumulation of temp files in `data/processed/`.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    suffix = path.suffix.lower()
    is_mp3 = suffix == ".mp3"

    try:
        # Keep WAV behavior unchanged and fast-path most inputs with librosa.
        y, sr = librosa.load(
            str(path),
            sr=target_sr,
            mono=True,
            duration=max_seconds,
        )
        if y is not None and len(y) > 0:
            return y.astype(np.float32), sr
    except Exception:
        if not is_mp3:
            # For non-MP3 files, we don't attempt extra conversion.
            raise

    if not is_mp3:
        # librosa loaded but returned empty (or duration too short).
        return np.zeros(target_sr, dtype=np.float32), target_sr

    tmp_wav_path: str | None = None
    try:
        # Create a temp WAV file to hold the MP3->WAV conversion.
        fd, tmp_wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        # pydub uses ffmpeg under the hood for mp3 decoding.
        # If ffmpeg isn't installed, AudioSegment.from_mp3 will raise.
        audio_seg = AudioSegment.from_mp3(str(path))
        audio_seg = audio_seg.set_frame_rate(target_sr)
        audio_seg.export(tmp_wav_path, format="wav")

        y, sr = librosa.load(
            tmp_wav_path,
            sr=target_sr,
            mono=True,
            duration=max_seconds,
        )
        if y is None or len(y) == 0:
            return np.zeros(target_sr, dtype=np.float32), target_sr
        return y.astype(np.float32), sr
    except Exception as e:
        raise RuntimeError(f"Could not decode MP3 audio via librosa or pydub: {path}") from e
    finally:
        # Best-effort cleanup.
        if tmp_wav_path:
            try:
                os.remove(tmp_wav_path)
            except OSError:
                pass


@dataclass(frozen=True)
class AudioProcessor:
    target_sr: int = 16000
    max_seconds: int = 30

    def load(self, path: str | Path) -> tuple[np.ndarray, int]:
        """
        Load and resample audio (supports WAV and MP3).
        """
        return load_audio(path, target_sr=self.target_sr, max_seconds=self.max_seconds)

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
