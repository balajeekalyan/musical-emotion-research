"""Lyric text embeddings for the RQ4 sub-study (voice vs words).

Only the 2,075 MERGE tracks that have lyrics. Uses the same sentence encoder as the prior
study (all-mpnet-base-v2) so the text channel is directly comparable across the two papers.

We store ONLY the derived embeddings, never the lyric text itself - MERGE's lyrics are not
ours to redistribute, and an embedding is not a reconstruction of the source text.

Output: outputs/lyrics_features.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
MODEL = "sentence-transformers/all-mpnet-base-v2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(args.threads)

    man = pd.read_parquet(OUT / "manifest.parquet",
                          columns=["song_id", "lyrics_text", "has_lyrics", "lyrics_word_count"])
    sub = man[man["has_lyrics"] & man["lyrics_text"].notna()].copy()
    sub["lyrics_text"] = sub["lyrics_text"].astype(str).str.strip()
    sub = sub[sub["lyrics_text"].str.len() > 0]
    print(f"tracks with usable lyrics: {len(sub)}")

    model = SentenceTransformer(MODEL)
    emb = model.encode(
        sub["lyrics_text"].tolist(),
        batch_size=args.batch, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
    ).astype(np.float32)
    print(f"embeddings: {emb.shape}")

    out = pd.DataFrame({
        "song_id": sub["song_id"].to_numpy(),
        "lyrics_sbert": list(emb),
        "lyrics_word_count": sub["lyrics_word_count"].to_numpy(),
    })
    path = OUT / "lyrics_features.parquet"
    out.to_parquet(path, index=False)
    print(f"wrote {path} rows={len(out)} dim={emb.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
