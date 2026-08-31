"""Phase 1 - build the study manifest.

Reuses the prior study's merged_dataset.parquet purely as an ID-alignment source
(it already resolves MERGE's three ID-linking schemes), then:
  * attaches both official MERGE tvt splits,
  * verifies every audio file is present and decodable,
  * audits the official splits for artist leakage,
  * derives an artist-disjoint split as a leakage-free alternative,
  * records where each stem will be written.

Output: outputs/manifest.parquet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
# Prior study checkout, holding merged_dataset.parquet (the ID-alignment source)
# and the MERGE distribution with its tvt_dataframes. Defaults to a sibling of
# this repository; override with PRIOR_STUDY_ROOT if it lives elsewhere.
PRIOR = Path(os.environ.get("PRIOR_STUDY_ROOT", ROOT.parent / "music_research"))
OUT = ROOT / "outputs"
STEMS = ROOT / "stems"

MERGED = PRIOR / "outputs" / "merged_dataset.parquet"
TVT = PRIOR / "MERGE_Audio_Balanced" / "tvt_dataframes"
STEM_NAMES = ["drums", "bass", "other", "vocals"]
SEED = 42

KEEP = [
    "song_id", "allmusic_id", "quadrant", "arousal", "valence",
    "artist", "title", "year", "audio_path",
    "lyrics_text", "lyrics_word_count", "has_lyrics", "is_multimodal",
]


def load_base() -> pd.DataFrame:
    df = pd.read_parquet(MERGED, columns=KEEP)
    print(f"base rows                : {len(df)}")
    assert df["song_id"].is_unique, "song_id must be unique"
    return df


def attach_official_splits(df: pd.DataFrame) -> pd.DataFrame:
    for scheme, folder in [("70_15_15", "tvt_70_15_15"), ("40_30_30", "tvt_40_30_30")]:
        mapping: dict[str, str] = {}
        for name, label in [("train", "train"), ("validate", "val"), ("test", "test")]:
            path = TVT / folder / f"tvt_{scheme}_{name}_audio_balanced.csv"
            songs = pd.read_csv(path)["Song"].astype(str)
            mapping.update(dict.fromkeys(songs, label))
        col = f"split_{scheme}"
        df[col] = df["song_id"].map(mapping)
        missing = int(df[col].isna().sum())
        counts = df[col].value_counts().to_dict()
        print(f"split {scheme:<9}        : {counts}  unmapped={missing}")
    return df


def verify_audio(df: pd.DataFrame) -> pd.DataFrame:
    """Every clip must decode; a file that fails here would silently poison a 5 h job."""
    durations, srs, channels, ok = [], [], [], []
    for path in df["audio_path"]:
        try:
            info = sf.info(path)
            durations.append(info.duration)
            srs.append(info.samplerate)
            channels.append(info.channels)
            ok.append(True)
        except Exception:
            durations.append(np.nan)
            srs.append(-1)
            channels.append(-1)
            ok.append(False)
    df["audio_duration"] = durations
    df["audio_sr"] = srs
    df["audio_channels"] = channels
    df["audio_ok"] = ok
    bad = int((~df["audio_ok"]).sum())
    print(f"audio decodable          : {len(df) - bad}/{len(df)}  (failed={bad})")
    print(f"  sample rates           : {df['audio_sr'].value_counts().to_dict()}")
    print(f"  channels               : {df['audio_channels'].value_counts().to_dict()}")
    print(
        f"  duration s             : min={df['audio_duration'].min():.2f} "
        f"median={df['audio_duration'].median():.2f} max={df['audio_duration'].max():.2f}"
    )
    if bad:
        print("  FAILED FILES:", df.loc[~df["audio_ok"], "song_id"].tolist()[:20])
    return df


def norm_artist(s: pd.Series) -> pd.Series:
    """Normalise for grouping only; the display name is left untouched."""
    a = s.fillna("").astype(str).str.strip().str.lower()
    a = a.str.replace(r"^the\s+", "", regex=True)
    a = a.str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    return a


def audit_artist_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """An emotion model can shortcut via artist identity if the same artist spans splits."""
    df["artist_key"] = norm_artist(df["artist"])
    unknown = df["artist_key"] == ""
    # unknown artists become singleton groups so they can never create false leakage
    df.loc[unknown, "artist_key"] = "unknown__" + df.loc[unknown, "song_id"].astype(str)
    n_artists = df["artist_key"].nunique()
    print(f"\nartists                  : {n_artists} unique over {len(df)} tracks "
          f"(unknown={int(unknown.sum())})")
    sizes = df["artist_key"].value_counts()
    print(f"  tracks per artist      : median={sizes.median():.0f} max={sizes.max()} "
          f"artists with >1 track={(sizes > 1).sum()}")

    for scheme in ["70_15_15", "40_30_30"]:
        col = f"split_{scheme}"
        sub = df.dropna(subset=[col])
        per = sub.groupby("artist_key")[col].nunique()
        spanning = set(per[per > 1].index)
        test = sub[sub[col] == "test"]
        leaked = test["artist_key"].isin(spanning).sum()
        print(
            f"  {scheme}: artists spanning splits={len(spanning)}  "
            f"test tracks sharing an artist with train/val={leaked}/{len(test)} "
            f"({100 * leaked / max(len(test), 1):.1f}%)"
        )
    return df


def artist_disjoint_split(df: pd.DataFrame) -> pd.DataFrame:
    """70/15/15 by artist group, keeping quadrant balance as even as possible."""
    y = df["quadrant"].to_numpy()
    groups = df["artist_key"].to_numpy()

    # 20 folds -> 14 train / 3 val / 3 test reproduces the official 70/15/15 ratio
    # while keeping every artist wholly inside one split
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    folds = np.empty(len(df), dtype=int)
    for i, (_, idx) in enumerate(sgkf.split(df, y, groups)):
        folds[idx] = i
    split = np.where(folds < 14, "train", np.where(folds < 17, "val", "test"))
    df["split_artist_disjoint"] = split

    per = df.groupby("artist_key")["split_artist_disjoint"].nunique()
    assert (per == 1).all(), "artist-disjoint split leaked an artist across splits"
    print(f"\nartist-disjoint split    : {pd.Series(split).value_counts().to_dict()}")
    xt = pd.crosstab(df["split_artist_disjoint"], df["quadrant"])
    print(xt.to_string())
    return df


def add_stem_paths(df: pd.DataFrame) -> pd.DataFrame:
    df["stem_dir"] = df["song_id"].apply(lambda s: str(STEMS / str(s)))
    for stem in STEM_NAMES:
        df[f"stem_{stem}_path"] = df["song_id"].apply(
            lambda s, k=stem: str(STEMS / str(s) / f"{k}.flac")
        )
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_base()
    df = attach_official_splits(df)
    df = verify_audio(df)
    df = audit_artist_leakage(df)
    df = artist_disjoint_split(df)
    df = add_stem_paths(df)

    print(f"\nquadrant balance         : {df['quadrant'].value_counts().sort_index().to_dict()}")
    print(f"tracks with lyrics       : {int(df['has_lyrics'].sum())}")

    path = OUT / "manifest.parquet"
    df.to_parquet(path, index=False)
    print(f"\nwrote {path}  shape={df.shape}")
    return 0 if df["audio_ok"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
