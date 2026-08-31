"""Phase 2 - separate every clip into drums/bass/other/vocals with Demucs hdemucs_mmi.

Runs ~5 h on this CPU-only machine, so it is resumable: completed songs are detected
on disk and skipped. Safe to interrupt and restart at any point.

Per song we write stems/<song_id>/{drums,bass,other,vocals}.flac and one log row
recording stem energies, the reconstruction error, and any peak rescaling applied.

Scaling note: Demucs' four stems sum to the input by construction. If any stem peaks
above full scale we scale ALL stems of that song by one shared factor, so the sum still
reconstructs the mix exactly up to that factor. The factor is logged.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
STEMS = ROOT / "stems"
LOG_JSONL = OUT / "separation_log.jsonl"

MODEL_NAME = "hdemucs_mmi"
OVERLAP = 0.25
SHIFTS = 0
SUBTYPE = "PCM_24"
STEM_NAMES = ["drums", "bass", "other", "vocals"]


def load_audio(path: str, sr: int) -> torch.Tensor:
    """soundfile decode (torchaudio.load needs TorchCodec as of 2.11) -> stereo @ sr."""
    data, orig_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(np.ascontiguousarray(data.T))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def song_done(song_id: str) -> bool:
    d = STEMS / str(song_id)
    return all((d / f"{s}.flac").exists() for s in STEM_NAMES)


def separate_one(model, wav: torch.Tensor) -> torch.Tensor:
    """Returns [n_sources, channels, samples] in the original signal scale."""
    from demucs.apply import apply_model

    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    wav_n = (wav - mean) / (std + 1e-8)
    with torch.no_grad():
        out = apply_model(
            model, wav_n[None], device="cpu", shifts=SHIFTS,
            split=True, overlap=OVERLAP, progress=False,
        )[0]
    return out * (std + 1e-8) + mean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process at most N songs")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from demucs.pretrained import get_model

    model = get_model(MODEL_NAME)
    model.eval()
    sr = model.samplerate
    sources = list(model.sources)
    assert sorted(sources) == sorted(STEM_NAMES), f"unexpected sources {sources}"

    df = pd.read_parquet(OUT / "manifest.parquet", columns=["song_id", "audio_path", "quadrant"])
    todo = [r for r in df.itertuples(index=False) if not song_done(r.song_id)]
    n_done = len(df) - len(todo)
    if args.limit:
        todo = todo[: args.limit]

    print(f"model={MODEL_NAME} sr={sr} sources={sources} threads={args.threads}")
    print(f"total={len(df)}  already done={n_done}  to process={len(todo)}")

    STEMS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    n_fail = 0
    with open(LOG_JSONL, "a", encoding="utf-8") as log:
        for i, row in enumerate(todo, 1):
            sid = str(row.song_id)
            t0 = time.perf_counter()
            try:
                wav = load_audio(row.audio_path, sr)
                est = separate_one(model, wav)

                mix_np = wav.numpy()
                est_np = est.numpy()

                # one shared rescale keeps stem sum == mix (up to the factor)
                peak = float(max(np.abs(est_np).max(), np.abs(mix_np).max()))
                scale = 0.999 / peak if peak > 0.999 else 1.0
                est_np = est_np * scale

                recon = est_np.sum(axis=0)
                recon_err = float(np.abs(recon - mix_np * scale).max())

                d = STEMS / sid
                d.mkdir(parents=True, exist_ok=True)
                rec = {
                    "song_id": sid,
                    "quadrant": row.quadrant,
                    "sr": sr,
                    "n_samples": int(mix_np.shape[1]),
                    "duration": float(mix_np.shape[1] / sr),
                    "scale": scale,
                    "recon_max_abs_err": recon_err,
                    "rms_mix": rms(mix_np * scale),
                }
                for j, name in enumerate(sources):
                    sf.write(d / f"{name}.flac", est_np[j].T, sr, subtype=SUBTYPE)
                    rec[f"rms_{name}"] = rms(est_np[j])
                rec["sec"] = time.perf_counter() - t0
                rec["ok"] = True
            except Exception as exc:
                n_fail += 1
                rec = {
                    "song_id": sid, "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "sec": time.perf_counter() - t0,
                }
            log.write(json.dumps(rec) + "\n")
            log.flush()

            if i % 10 == 0 or i == len(todo):
                el = time.perf_counter() - t_start
                rate = el / i
                eta = rate * (len(todo) - i) / 3600
                print(
                    f"[{i}/{len(todo)}] {rate:.2f}s/song  elapsed={el/3600:.2f}h  "
                    f"eta={eta:.2f}h  fail={n_fail}",
                    flush=True,
                )

    consolidate()
    return 0


def consolidate() -> None:
    if not LOG_JSONL.exists():
        return
    rows = [json.loads(l) for l in LOG_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    log = pd.DataFrame(rows).drop_duplicates(subset="song_id", keep="last")
    path = OUT / "separation_log.parquet"
    log.to_parquet(path, index=False)
    ok = log["ok"].sum() if "ok" in log else 0
    print(f"\nwrote {path}  rows={len(log)}  ok={ok}  failed={len(log) - ok}")
    if "recon_max_abs_err" in log:
        e = log["recon_max_abs_err"].dropna()
        if len(e):
            print(f"reconstruction max-abs-err: median={e.median():.2e} max={e.max():.2e}")


if __name__ == "__main__":
    raise SystemExit(main())
