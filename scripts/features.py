"""Hand-crafted acoustic feature extraction (134 dims).

Feature set mirrors the prior study's so that the two papers' audio representations are
comparable. Every condition gets exactly this budget (control C5) - no condition may win
by having more features.
"""
from __future__ import annotations

import numpy as np
import librosa

SR_HANDCRAFTED = 22050
N_MFCC = 40
N_FFT = 2048
HOP = 512

FEATURE_NAMES: list[str] = (
    [f"mfcc{i}_mean" for i in range(N_MFCC)]
    + [f"mfcc{i}_std" for i in range(N_MFCC)]
    + ["centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std",
       "rolloff_mean", "rolloff_std", "flux_mean", "flux_std"]
    + [f"chroma{i}_mean" for i in range(12)]
    + [f"chroma{i}_std" for i in range(12)]
    + [f"tonnetz{i}_mean" for i in range(6)]
    + [f"tonnetz{i}_std" for i in range(6)]
    + ["tempo", "beat_count", "onset_mean", "onset_std", "zcr_mean", "zcr_std"]
    + ["rms_mean", "rms_std", "dynamic_range_db", "crest_factor"]
)
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 134, N_FEATURES


def _ms(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x.mean(axis=1), x.std(axis=1)


def handcrafted(y: np.ndarray, sr: int = SR_HANDCRAFTED) -> np.ndarray:
    """y: mono float32. Returns a 134-d float32 vector aligned to FEATURE_NAMES."""
    y = np.asarray(y, dtype=np.float32)
    if y.size < N_FFT or not np.any(np.isfinite(y)) or not np.any(y):
        return np.zeros(N_FEATURES, dtype=np.float32)

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(S=S**2, sr=sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)
    mfcc_m, mfcc_s = _ms(mfcc)

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    flux = librosa.onset.onset_strength(S=librosa.power_to_db(S**2), sr=sr)[None, :]

    # one CQT chroma serves both the chroma block and tonnetz
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    chroma_m, chroma_s = _ms(chroma)
    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sr)
    ton_m, ton_s = _ms(tonnetz)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0])

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP)
    rms = librosa.feature.rms(S=S)

    r = rms.ravel()
    nz = r[r > 0]
    if nz.size:
        db = 20.0 * np.log10(nz)
        dyn = float(np.percentile(db, 95) - np.percentile(db, 5))
    else:
        dyn = 0.0
    crest = float(np.abs(y).max() / (np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12))

    vec = np.concatenate([
        mfcc_m, mfcc_s,
        [centroid.mean(), centroid.std(), bandwidth.mean(), bandwidth.std(),
         rolloff.mean(), rolloff.std(), flux.mean(), flux.std()],
        chroma_m, chroma_s, ton_m, ton_s,
        [tempo, float(len(beats)), onset_env.mean(), onset_env.std(),
         zcr.mean(), zcr.std()],
        [r.mean(), r.std(), dyn, crest],
    ]).astype(np.float32)

    assert vec.shape == (N_FEATURES,), vec.shape
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
