"""Shared definitions: how the stem conditions are built, and loudness handling.

Kept separate from the extractors so that Phase 3, the controls in Phase 5 and any
qualitative rendering all construct conditions from exactly one implementation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

STEM_NAMES = ["drums", "bass", "other", "vocals"]

# condition name -> stems summed to build it. `mix` and `residual` are special-cased.
CONDITIONS: dict[str, tuple[str, ...] | None] = {
    "mix": None,                                            # original recording
    "vocals": ("vocals",),                                  # sufficiency
    "drums": ("drums",),
    "bass": ("bass",),
    "other": ("other",),
    "accompaniment": ("drums", "bass", "other"),            # necessity: no voice
    "no_drums": ("bass", "other", "vocals"),
    "no_bass": ("drums", "other", "vocals"),
    "no_other": ("drums", "bass", "vocals"),
    "remix": ("drums", "bass", "other", "vocals"),          # control
    "residual": None,                                       # control: mix - remix
}

# C3 artifact null: spectral-shuffled versions of the mix and the four single stems.
# Only the sufficiency conditions need the null - they carry the claims most at risk
# from separator artifacts - plus the mix as the reference point.
SHUFFLE_BASE = ("mix", "vocals", "drums", "bass", "other")
SHUFFLE_PREFIX = "shuffle_"
SHUFFLE_N_FFT = 2048
SHUFFLE_HOP = 512

TARGET_LUFS = -23.0          # EBU R128
SILENCE_LUFS = -70.0         # below this we refuse to normalise
MAX_GAIN_DB = 40.0           # never amplify a near-silent stem beyond this


def load_stems(stem_dir: str | Path) -> dict[str, np.ndarray]:
    """Returns {stem: float32 array [channels, samples]}."""
    stem_dir = Path(stem_dir)
    out = {}
    for name in STEM_NAMES:
        data, _ = sf.read(stem_dir / f"{name}.flac", dtype="float32", always_2d=True)
        out[name] = np.ascontiguousarray(data.T)
    return out


def build_conditions(stems: dict[str, np.ndarray], mix: np.ndarray) -> dict[str, np.ndarray]:
    """`mix` must already be on the same scale as the stems (see separation `scale`)."""
    conds: dict[str, np.ndarray] = {}
    n = min(mix.shape[1], min(s.shape[1] for s in stems.values()))
    mix = mix[:, :n]
    stems = {k: v[:, :n] for k, v in stems.items()}

    for name, parts in CONDITIONS.items():
        if name == "mix":
            conds[name] = mix
        elif name == "residual":
            conds[name] = mix - sum(stems[s] for s in STEM_NAMES)
        else:
            conds[name] = sum(stems[s] for s in parts)
    return conds


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0) if x.ndim == 2 else x


def measure_loudness(x_stereo: np.ndarray, sr: int, meter: pyln.Meter | None = None) -> float:
    """Integrated LUFS. Returns -inf for digital silence or clips shorter than the gate."""
    if not np.any(x_stereo):
        return float("-inf")
    if x_stereo.shape[1] / sr < 0.5:
        return float("-inf")
    meter = meter or pyln.Meter(sr)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(meter.integrated_loudness(x_stereo.T))


def normalise(x_stereo: np.ndarray, lufs: float) -> tuple[np.ndarray, float, bool]:
    """Loudness-normalise to TARGET_LUFS.

    Returns (audio, gain_db_applied, is_silent). Near-silent conditions are left
    untouched rather than amplified by an absurd factor, and flagged so downstream
    analysis can exclude them.
    """
    if not np.isfinite(lufs) or lufs < SILENCE_LUFS:
        return x_stereo, 0.0, True
    gain_db = float(np.clip(TARGET_LUFS - lufs, -np.inf, MAX_GAIN_DB))
    return x_stereo * (10.0 ** (gain_db / 20.0)), gain_db, False


def shuffle_seed(song_id: str, condition: str) -> int:
    """Deterministic per (song, condition), so resumed shards and re-runs agree."""
    import zlib
    return zlib.crc32(f"{song_id}/{condition}".encode())


def spectral_shuffle(x_stereo: np.ndarray, seed: int) -> np.ndarray:
    """C3 artifact null: permute the magnitude STFT's frequency bins independently
    within every frame, give every bin a random phase, and resynthesise.

    Per-frame spectral energy is preserved, so the energy envelope survives at STFT
    frame resolution (~12 ms at 44.1 kHz) - loudness dynamics and coarse rhythm remain.
    Pitch, harmony, timbre and every other piece of fine spectral structure are
    destroyed. Whatever predictive power survives this is trivial energy information,
    not musical content.
    """
    import librosa

    rng = np.random.default_rng(seed)
    out = np.empty_like(x_stereo)
    for ch in range(x_stereo.shape[0]):
        S = librosa.stft(x_stereo[ch], n_fft=SHUFFLE_N_FFT, hop_length=SHUFFLE_HOP)
        mag = np.abs(S)
        order = np.argsort(rng.random(mag.shape), axis=0)   # fresh permutation per frame
        mag = np.take_along_axis(mag, order, axis=0)
        phase = np.exp(2j * np.pi * rng.random(mag.shape))
        y = librosa.istft(mag * phase, hop_length=SHUFFLE_HOP, n_fft=SHUFFLE_N_FFT,
                          length=x_stereo.shape[1])
        # random phases make overlapping frames add incoherently, which costs a uniform
        # ~sqrt(overlap) amplitude factor; rescale so total energy matches the input and
        # the energy_share metadata stays comparable to the real conditions
        r_in, r_out = rms(x_stereo[ch]), rms(y)
        if r_out > 0:
            y = y * (r_in / r_out)
        out[ch] = y.astype(x_stereo.dtype, copy=False)
    return out


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
