"""Benchmark CPU separation models on a handful of MERGE clips.

No NVIDIA GPU on this machine, so throughput decides which model we can afford
across the full 3,232-clip corpus. Reports seconds per 30 s clip and the implied
wall-clock for the whole corpus.
"""
import os
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

warnings.filterwarnings("ignore")

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRIOR = Path(os.environ.get("PRIOR_STUDY_ROOT", ROOT.parent / "music_research"))
MERGED = PRIOR / "outputs" / "merged_dataset.parquet"
_df = pd.read_parquet(MERGED, columns=["song_id", "quadrant", "audio_path"])
# one clip per quadrant so the timing covers a range of material
CLIPS = (
    _df.sort_values("song_id").groupby("quadrant").head(1)["audio_path"].tolist()
)
N_CORPUS = len(_df)

torch.set_num_threads(12)


def load(path, sr):
    """torchaudio.load needs TorchCodec as of 2.11, so decode with soundfile."""
    data, orig_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(np.ascontiguousarray(data.T))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav


def bench_demucs(model_name, overlap):
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model(model_name)
    model.eval()
    sr = model.samplerate

    times = []
    for path in CLIPS:
        wav = load(path, sr)
        ref = wav.mean(0)
        wav_n = (wav - ref.mean()) / ref.std()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = apply_model(
                model, wav_n[None], device="cpu", shifts=0,
                split=True, overlap=overlap, progress=False,
            )
        times.append(time.perf_counter() - t0)
    return times, model.sources, sum(p.numel() for p in model.parameters())


def bench_torchaudio_hdemucs():
    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    model.eval()
    sr = bundle.sample_rate

    times = []
    for path in CLIPS:
        wav = load(path, sr)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(wav[None])
        times.append(time.perf_counter() - t0)
    return times, model.sources, sum(p.numel() for p in model.parameters())


def report(label, times, sources, nparams):
    mean = sum(times) / len(times)
    hours = mean * N_CORPUS / 3600
    print(
        f"{label:<34} {mean:7.2f}s/clip   corpus={hours:6.2f}h   "
        f"params={nparams/1e6:5.1f}M   sources={sources}"
    )
    return mean


if __name__ == "__main__":
    print(f"threads={torch.get_num_threads()}  clips={len(CLIPS)}  corpus={N_CORPUS}\n")
    results = {}

    for name, overlap in [
        ("htdemucs", 0.25),
        ("htdemucs", 0.10),
        ("hdemucs_mmi", 0.25),
        ("mdx_extra", 0.25),
    ]:
        try:
            times, sources, nparams = bench_demucs(name, overlap)
            results[f"{name}@ov{overlap}"] = report(
                f"demucs:{name} overlap={overlap}", times, sources, nparams
            )
        except Exception as exc:
            print(f"demucs:{name} overlap={overlap:<5} FAILED -> {type(exc).__name__}: {exc}")

    try:
        times, sources, nparams = bench_torchaudio_hdemucs()
        results["torchaudio_hdemucs"] = report(
            "torchaudio:HDEMUCS_HIGH_MUSDB+", times, sources, nparams
        )
    except Exception as exc:
        print(f"torchaudio:HDEMUCS FAILED -> {type(exc).__name__}: {exc}")

    print("\nfastest ->", min(results, key=results.get) if results else "none")
