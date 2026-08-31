"""Phase 3 - per-condition feature extraction.

Two stages, run separately because they have very different parallelism profiles:

  --stage handcrafted   librosa, CPU-bound per song  -> ProcessPoolExecutor
  --stage clap          torch, benefits from batching -> single process, batched

Both are resumable at shard granularity: a shard already on disk is skipped, so an
interrupted run resumes without redoing work.

CLAP note: laion/larger_clap_music has enable_fusion=False and a 10 s input window with
truncation='rand_trunc', i.e. the processor would take a RANDOM 10 s crop. That is
non-deterministic and discards two thirds of each clip. We instead cut each condition into
fixed non-overlapping 10 s windows, embed every window, and mean-pool - deterministic and
uses the whole excerpt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import (  # noqa: E402
    CONDITIONS, SHUFFLE_BASE, SHUFFLE_PREFIX, build_conditions, load_stems,
    measure_loudness, normalise, rms, shuffle_seed, spectral_shuffle, to_mono,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FEAT = OUT / "features"

SR_SEP = 44100
SR_CLAP = 48000
CLAP_WINDOW_S = 10
CLAP_MIN_TAIL_FRAC = 0.5     # keep a trailing partial window only if this full
SHARD = 100


def clap_windows(y: np.ndarray, win: int) -> list[np.ndarray]:
    """Cut a signal into fixed non-overlapping windows for CLAP.

    A trailing partial window is kept only if it is at least CLAP_MIN_TAIL_FRAC full.
    This matters more than it looks: 60 % of MERGE clips run a hair over 30 s, so a
    naive ceil() appends a fourth window holding ~0.06 s of audio and ~9.94 s of zero
    padding. Mean-pooling would then drag those songs' embeddings a quarter of the way
    toward silence while leaving the other 40 % untouched - a corpus-splitting artifact
    with no musical meaning. At least one window is always returned.
    """
    n_full = len(y) // win
    segs = [y[k * win:(k + 1) * win] for k in range(n_full)]
    tail = y[n_full * win:]
    if len(tail) >= CLAP_MIN_TAIL_FRAC * win:
        segs.append(np.pad(tail, (0, win - len(tail))))
    if not segs:                                  # clip shorter than one window
        segs = [np.pad(y, (0, max(0, win - len(y))))[:win]]
    return [np.ascontiguousarray(s, dtype=np.float32) for s in segs]


def load_separation_log() -> pd.DataFrame:
    """Prefer the append-only jsonl over the consolidated parquet.

    separate_stems.py only writes the parquet after finishing its whole loop, so an
    interrupted (or still-running) separation leaves the parquet stale. Reading the
    jsonl means Phase 3 always sees every song separated so far instead of silently
    extracting features for a handful.
    """
    jsonl = OUT / "separation_log.jsonl"
    if jsonl.exists():
        rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        if rows:
            return pd.DataFrame(rows).drop_duplicates(subset="song_id", keep="last")
    return pd.read_parquet(OUT / "separation_log.parquet")


def songs_available() -> pd.DataFrame:
    """Every song that has been separated successfully, in a stable order.

    Shared by the extraction stages and by assemble(), so that "what should exist" is
    defined in exactly one place and the completeness check cannot drift from the work.
    """
    man = pd.read_parquet(OUT / "manifest.parquet")
    log = load_separation_log()
    log = log[log["ok"]] if "ok" in log else log
    return (
        man.merge(log[["song_id", "scale"]], on="song_id", how="inner")
        [["song_id", "audio_path", "stem_dir", "scale", "quadrant"]]
        .sort_values("song_id")
        .reset_index(drop=True)
    )


def _shard_done(path: Path, shard: pd.DataFrame) -> bool:
    """A shard counts as done only if it actually covers every song assigned to it.

    Existence alone is not enough: a --limit smoke run writes a shard_0000 holding just a
    few songs, and a later full run would skip it and silently drop the rest of that shard.
    """
    if not path.exists():
        return False
    try:
        have = set(pd.read_parquet(path, columns=["song_id"])["song_id"].astype(str))
    except Exception:
        return False
    return set(shard["song_id"].astype(str)) <= have


def _pending(outdir: Path, shards: list[pd.DataFrame]) -> list[tuple[int, pd.DataFrame]]:
    return [(i, s) for i, s in enumerate(shards)
            if not _shard_done(outdir / f"shard_{i:04d}.parquet", s)]


def _stage_dir(stage: str, loudnorm: bool, shuffle: bool) -> str:
    if shuffle:
        return f"{stage}_shuffle"
    return stage if loudnorm else f"{stage}_raw"


def _load_song_conditions(row, loudnorm: bool = True,
                          shuffle: bool = False) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Rebuild every condition for one song, plus per-condition meta.

    With loudnorm=False the conditions are returned at their natural level. That is the
    C1 control: stems differ hugely in level, so any conclusion that changes between the
    two passes is loudness-driven and must be reported as such rather than hidden.
    The metadata (lufs, energy_share, silent) is measured pre-normalisation either way,
    so it is identical across the two passes and stays comparable.
    """
    import soundfile as sf
    import librosa
    import pyloudnorm as pyln

    data, sr0 = sf.read(row.audio_path, dtype="float32", always_2d=True)
    mix = np.ascontiguousarray(data.T)
    if mix.shape[0] == 1:
        mix = mix.repeat(2, axis=0)
    elif mix.shape[0] > 2:
        mix = mix[:2]
    if sr0 != SR_SEP:
        mix = librosa.resample(mix, orig_sr=sr0, target_sr=SR_SEP)
    mix = mix * float(row.scale)

    stems = load_stems(row.stem_dir)
    conds = build_conditions(stems, mix)
    mix_rms = rms(conds["mix"])
    if shuffle:
        # C3 artifact null: envelope-preserving, structure-destroying versions of the
        # mix and the four single stems. energy_share stays relative to the REAL mix,
        # and the loudness normalisation below is identical to the main pass (C5).
        conds = {
            SHUFFLE_PREFIX + name: spectral_shuffle(
                conds[name], shuffle_seed(str(row.song_id), name))
            for name in SHUFFLE_BASE
        }

    meter = pyln.Meter(SR_SEP)
    audio, meta = {}, {}
    for name, x in conds.items():
        lufs = measure_loudness(x, SR_SEP, meter)
        xn, gain_db, silent = normalise(x, lufs)
        if not loudnorm:
            xn, gain_db = x, 0.0
        audio[name] = to_mono(xn)
        r = rms(x)
        meta[name] = {
            "lufs": lufs if np.isfinite(lufs) else np.nan,
            "gain_db": gain_db,
            "silent": silent,
            "rms_raw": r,
            "energy_share": (r / mix_rms) if mix_rms > 0 else np.nan,
        }
    return audio, meta


# ---------------------------------------------------------------- handcrafted

def _handcrafted_song(row_dict: dict) -> list[dict]:
    import librosa
    from features import handcrafted, SR_HANDCRAFTED

    row = argparse.Namespace(**row_dict)
    try:
        # loudnorm/shuffle travel inside the payload: Windows spawns workers, so a
        # module-level global set in main() would not reach them
        audio, meta = _load_song_conditions(row, loudnorm=row_dict.get("loudnorm", True),
                                            shuffle=row_dict.get("shuffle", False))
    except Exception as exc:
        return [{"song_id": row.song_id, "condition": None,
                 "error": f"{type(exc).__name__}: {exc}"}]

    rows = []
    for name, mono in audio.items():
        y = librosa.resample(mono, orig_sr=SR_SEP, target_sr=SR_HANDCRAFTED)
        rows.append({
            "song_id": row.song_id, "condition": name,
            "handcrafted": handcrafted(y).astype(np.float32),
            **meta[name], "error": None,
        })
    return rows


def run_handcrafted(df: pd.DataFrame, workers: int, loudnorm: bool = True,
                    shuffle: bool = False) -> None:
    from concurrent.futures import ProcessPoolExecutor

    outdir = FEAT / _stage_dir("handcrafted", loudnorm, shuffle)
    outdir.mkdir(parents=True, exist_ok=True)
    shards = [df.iloc[i:i + SHARD] for i in range(0, len(df), SHARD)]
    todo = _pending(outdir, shards)
    print(f"handcrafted: {len(shards)} shards, {len(todo)} to do, workers={workers} "
          f"loudnorm={loudnorm} shuffle={shuffle}", flush=True)

    t0 = time.perf_counter()
    for n, (i, shard) in enumerate(todo, 1):
        payload = [r | {"loudnorm": loudnorm, "shuffle": shuffle}
                   for r in shard.to_dict("records")]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_handcrafted_song, payload, chunksize=1))
        rows = [r for sub in results for r in sub]
        pd.DataFrame(rows).to_parquet(outdir / f"shard_{i:04d}.parquet", index=False)
        el = time.perf_counter() - t0
        print(f"  shard {i:04d} [{n}/{len(todo)}] {el/n:.1f}s/shard "
              f"eta={(len(todo)-n)*el/n/3600:.2f}h", flush=True)


# ----------------------------------------------------------------------- CLAP

def _prefetch(rows, fn, depth: int = 2):
    """Yield (row, result, exc), preparing `depth` rows ahead on a worker thread.

    Audio prep (FLAC decode, condition sums, loudness metering, resampling) is ~3.4 s
    per song of mostly GIL-releasing C and numpy work, during which the torch threads
    would otherwise sit idle waiting for the next batch. Overlapping it with the forward
    pass hides most of that. Queue depth is bounded because one song's conditions are
    ~58 MB.
    """
    from queue import Queue
    from threading import Thread

    q: Queue = Queue(maxsize=depth)
    done = object()

    def worker():
        for r in rows:
            try:
                q.put((r, fn(r), None))
            except Exception as exc:                     # noqa: BLE001 - reported per row
                q.put((r, None, exc))
        q.put(done)

    t = Thread(target=worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is done:
            return
        yield item


def run_clap(df: pd.DataFrame, threads: int, batch: int, loudnorm: bool = True,
             shuffle: bool = False) -> None:
    import torch
    import librosa
    from transformers import ClapModel, ClapProcessor

    torch.set_num_threads(threads)
    model = ClapModel.from_pretrained("laion/larger_clap_music").eval()
    proc = ClapProcessor.from_pretrained("laion/larger_clap_music")
    win = CLAP_WINDOW_S * SR_CLAP

    outdir = FEAT / _stage_dir("clap", loudnorm, shuffle)
    outdir.mkdir(parents=True, exist_ok=True)
    shards = [df.iloc[i:i + SHARD] for i in range(0, len(df), SHARD)]
    todo = _pending(outdir, shards)
    print(f"clap: {len(shards)} shards, {len(todo)} to do, threads={threads} batch={batch} "
          f"loudnorm={loudnorm} shuffle={shuffle}", flush=True)

    t0 = time.perf_counter()
    for n, (i, shard) in enumerate(todo, 1):
        rows = []
        # Windows are mean-pooled per (song, condition) via a running sum, and the clip
        # buffer is flushed as soon as it fills a batch. Buffering a whole 100-song shard
        # instead would hold ~4400 x 10 s x 48 kHz float32 = ~8 GB of audio at once.
        clips: list[np.ndarray] = []
        keys: list[tuple] = []
        acc: dict[tuple, np.ndarray] = {}
        cnt: dict[tuple, int] = {}

        def flush(force: bool = False) -> None:
            while clips and (force or len(clips) >= batch):
                take = min(batch, len(clips))
                inp = proc(audios=clips[:take], sampling_rate=SR_CLAP,
                           return_tensors="pt", padding=True)
                with torch.no_grad():
                    emb = model.get_audio_features(**inp).numpy()
                for k, e in zip(keys[:take], emb):
                    if k in acc:
                        acc[k] += e
                        cnt[k] += 1
                    else:
                        acc[k] = e.astype(np.float64).copy()
                        cnt[k] = 1
                del clips[:take]
                del keys[:take]

        prepared = _prefetch(
            list(shard.itertuples(index=False)),
            lambda r: _load_song_conditions(r, loudnorm=loudnorm, shuffle=shuffle),
        )
        for row, res, exc in prepared:
            if exc is not None:
                rows.append({"song_id": row.song_id, "condition": None,
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            audio, meta = res
            for name, mono in audio.items():
                y = librosa.resample(mono, orig_sr=SR_SEP, target_sr=SR_CLAP)
                for seg in clap_windows(y, win):
                    clips.append(seg)
                    keys.append((row.song_id, name))
                rows.append({"song_id": row.song_id, "condition": name,
                             **meta[name], "error": None})
            del audio
            flush()
        flush(force=True)

        pooled = {k: (acc[k] / cnt[k]).astype(np.float32) for k in acc}
        for r in rows:
            r["clap"] = pooled.get((r["song_id"], r["condition"]))

        pd.DataFrame(rows).to_parquet(outdir / f"shard_{i:04d}.parquet", index=False)
        el = time.perf_counter() - t0
        print(f"  shard {i:04d} [{n}/{len(todo)}] {el/n:.1f}s/shard "
              f"eta={(len(todo)-n)*el/n/3600:.2f}h", flush=True)


# ------------------------------------------------------------------ assemble

def assemble(loudnorm: bool = True, allow_partial: bool = False,
             shuffle: bool = False) -> int:
    """Merge the shard files into the single Phase 3 output.

    Refuses to write an incomplete table unless explicitly told to. A partial feature
    table is the most dangerous artifact this pipeline can produce: it loads fine, has
    the right columns, and quietly drops songs or an entire feature block, so Phase 4
    would train on it and report plausible-looking numbers. An interrupted stage must
    fail loudly here rather than hand downstream work a silent subset.
    """
    parts = {}
    for stage in ["handcrafted", "clap"]:
        sub = _stage_dir(stage, loudnorm, shuffle)
        files = sorted((FEAT / sub).glob("shard_*.parquet"))
        if not files:
            print(f"assemble: no shards for {stage}")
            continue
        parts[stage] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        print(f"assemble: {stage} rows={len(parts[stage])} "
              f"songs={parts[stage]['song_id'].nunique()}")

    expected = set(songs_available()["song_id"].astype(str))
    problems = []
    for stage in ["handcrafted", "clap"]:
        if stage not in parts:
            problems.append(f"{stage}: no shards on disk")
            continue
        got = set(parts[stage]["song_id"].astype(str))
        missing = expected - got
        if missing:
            problems.append(f"{stage}: {len(missing)} of {len(expected)} songs missing "
                            f"(e.g. {sorted(missing)[:3]})")
        errs = int(parts[stage]["error"].notna().sum())
        if errs:
            problems.append(f"{stage}: {errs} rows carry an extraction error")

    if problems and not allow_partial:
        print("\nassemble: REFUSING to write - inputs are incomplete:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("\nRe-run the unfinished stage (it resumes), or pass --allow-partial if a\n"
              "deliberately partial table is what you want.", file=sys.stderr)
        return 1

    if not parts:
        return 1
    if len(parts) == 2:
        # keep handcrafted's copy of the shared per-condition metadata
        df = parts["handcrafted"].merge(
            parts["clap"][["song_id", "condition", "clap"]],
            on=["song_id", "condition"], how="outer",
        )
    else:
        df = next(iter(parts.values()))

    df = df[df["condition"].notna()].reset_index(drop=True)
    suffix = "_shuffle" if shuffle else ("" if loudnorm else "_raw")
    if problems:
        suffix += "_partial"          # never overwrite the real output with a subset
    path = OUT / f"stem_features{suffix}.parquet"
    df.to_parquet(path, index=False)
    print(f"wrote {path} shape={df.shape} conditions={df['condition'].nunique()} "
          f"songs={df['song_id'].nunique()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["handcrafted", "clap", "assemble"], required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)   # 0.21 s/clip vs 0.29 at batch 8
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-loudnorm", dest="loudnorm", action="store_false",
                    help="C1 control: extract at natural level, into *_raw outputs")
    ap.add_argument("--shuffle", action="store_true",
                    help="C3 control: spectral-shuffle nulls of mix + single stems, "
                         "into *_shuffle outputs")
    ap.add_argument("--allow-partial", action="store_true",
                    help="assemble: write even if shards are incomplete (-> *_partial)")
    args = ap.parse_args()

    if args.shuffle and not args.loudnorm:
        print("--shuffle implies the loudness-normalised pass; drop --no-loudnorm",
              file=sys.stderr)
        return 1

    if args.stage == "assemble":
        return assemble(args.loudnorm, args.allow_partial, args.shuffle)

    df = songs_available()
    if args.limit:
        df = df.head(args.limit)
    n_conds = len(SHUFFLE_BASE) if args.shuffle else len(CONDITIONS)
    print(f"songs available: {len(df)}  conditions: {n_conds}")

    if args.stage == "handcrafted":
        run_handcrafted(df, args.workers, args.loudnorm, args.shuffle)
    else:
        run_clap(df, args.threads, args.batch, args.loudnorm, args.shuffle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
