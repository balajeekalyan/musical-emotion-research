"""Phase 7 - RQ4/H4: voice vs words.

Restricted to the 2,075 MERGE tracks that have both lyrics and a vocal stem. Compares
three channels for predicting arousal/valence:
  audio  - vocal-stem acoustic features alone (how it is sung), `both` featset
  text   - SBERT lyric embeddings alone (what is sung)
  both   - the two channels concatenated

H4 falsification test (rq4_combo_vs_best.csv): H4 is falsified if the combined channel
fails to beat the better of the two single channels, paired-bootstrapped on the same
tracks. Error correlation (rq4_error_correlation.csv): low correlation between the
audio-only and text-only residuals means the two channels fail on different tracks,
i.e. genuinely complementary information rather than two views of the same signal.
Divergent tracks (rq4_divergent_tracks.csv) is the qualitative "sung one way, written
another" pass - the test tracks where the two channels' predictions disagree most.

Reuses Phase 4's build_matrix/fit_regression/boot_ci and Phase 5's paired_bootstrap
directly, so RQ4 numbers are produced the same way as the main results rather than by
a separate implementation.

Outputs (outputs/):
  rq4_results.csv              one row per (split, model, channel, target)
  rq4_test_predictions.parquet per-track test predictions for every channel
  rq4_combo_vs_best.csv        H4 test: both - max(audio, text), paired bootstrap
  rq4_error_correlation.csv    correlation of audio vs text residuals
  rq4_divergent_tracks.csv     top divergent tracks per (split, model, target)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_models import SPLITS, REG_TARGETS, build_matrix, fit_regression, boot_ci  # noqa: E402
from phase5_analysis import paired_bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

AUDIO_FEATSET = "both"   # handcrafted + CLAP, matching the main study's primary spec
CHANNELS = ["audio", "text", "both"]
TOP_N_DIVERGENT = 20


def load_rq4_data(featset: str = AUDIO_FEATSET) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    man = pd.read_parquet(OUT / "manifest.parquet")
    lyr = pd.read_parquet(OUT / "lyrics_features.parquet")
    feat = pd.read_parquet(OUT / "stem_features.parquet")
    feat = feat[(feat["condition"] == "vocals") & feat["song_id"].isin(set(lyr["song_id"]))]

    idx, X_audio = build_matrix(feat, "vocals", featset)
    sub = feat.loc[idx].merge(lyr, on="song_id", how="left") \
                       .merge(man[["song_id", "arousal", "valence", *SPLITS.values()]],
                              on="song_id", how="left")
    X_text = np.nan_to_num(np.stack(sub["lyrics_sbert"].to_numpy()).astype(np.float32),
                           nan=0.0, posinf=0.0, neginf=0.0)
    print(f"RQ4 subset: {len(sub)} tracks with vocal-stem audio + lyrics "
          f"(of {len(lyr)} lyrics tracks)")
    return sub.reset_index(drop=True), X_audio, X_text


def _wide(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return df.pivot_table(index="song_id", columns="channel", values=value, aggfunc="first")


def combo_vs_best(pdf: pd.DataFrame) -> pd.DataFrame:
    """H4: is the combined channel better than the better single channel?"""
    rows = []
    for key, g in pdf.groupby(["split", "model", "target"]):
        truth_w, pred_w = _wide(g, "y_true"), _wide(g, "y_pred")
        common = truth_w.index.intersection(pred_w.index)
        if not set(CHANNELS) <= set(pred_w.columns) or not len(common):
            continue
        truth = truth_w.loc[common, "audio"].to_numpy()
        audio_r2 = r2_score(truth, pred_w.loc[common, "audio"])
        text_r2 = r2_score(truth, pred_w.loc[common, "text"])
        best_single = "audio" if audio_r2 >= text_r2 else "text"
        d, lo, hi, sig = paired_bootstrap(
            truth, pred_w.loc[common, "both"].to_numpy(),
            pred_w.loc[common, best_single].to_numpy(), r2_score,
        )
        rows.append({
            "split": key[0], "model": key[1], "target": key[2],
            "audio_r2": audio_r2, "text_r2": text_r2, "best_single": best_single,
            "both_r2": r2_score(truth, pred_w.loc[common, "both"]),
            "diff_both_minus_best_single": d, "ci_lo": lo, "ci_hi": hi,
            "significant": sig, "h4_supported": bool(sig and d > 0), "n": len(common),
        })
    return pd.DataFrame(rows)


def error_correlation(pdf: pd.DataFrame) -> pd.DataFrame:
    """How correlated are the audio-only and text-only residuals?"""
    rows = []
    for key, g in pdf.groupby(["split", "model", "target"]):
        truth_w, pred_w = _wide(g, "y_true"), _wide(g, "y_pred")
        common = truth_w.index.intersection(pred_w.index)
        if not {"audio", "text"} <= set(pred_w.columns) or not len(common):
            continue
        truth = truth_w.loc[common, "audio"].to_numpy()
        err_audio = truth - pred_w.loc[common, "audio"].to_numpy()
        err_text = truth - pred_w.loc[common, "text"].to_numpy()
        r_p, p_p = pearsonr(err_audio, err_text)
        rows.append({
            "split": key[0], "model": key[1], "target": key[2],
            "pearson_r": float(r_p), "pearson_p": float(p_p),
            "spearman_r": float(spearmanr(err_audio, err_text).statistic), "n": len(common),
        })
    return pd.DataFrame(rows)


def divergent_tracks(pdf: pd.DataFrame, top_n: int = TOP_N_DIVERGENT) -> pd.DataFrame:
    """Qualitative pass: test tracks where the audio-only and text-only PREDICTIONS
    disagree most sharply - the "sung one way, written another" cases."""
    rows = []
    for key, g in pdf.groupby(["split", "model", "target"]):
        truth_w, pred_w = _wide(g, "y_true"), _wide(g, "y_pred")
        common = truth_w.index.intersection(pred_w.index)
        if not {"audio", "text"} <= set(pred_w.columns) or not len(common):
            continue
        sub = pd.DataFrame({
            "song_id": common,
            "y_true": truth_w.loc[common, "audio"].to_numpy(),
            "pred_audio": pred_w.loc[common, "audio"].to_numpy(),
            "pred_text": pred_w.loc[common, "text"].to_numpy(),
        })
        sub["divergence"] = (sub["pred_audio"] - sub["pred_text"]).abs()
        sub = sub.sort_values("divergence", ascending=False).head(top_n)
        sub["split"], sub["model"], sub["target"] = key
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    cols = ["split", "model", "target", "song_id", "y_true", "pred_audio", "pred_text", "divergence"]
    return pd.concat(rows, ignore_index=True)[cols]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ridge"], choices=["ridge", "gbm"])
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    df, X_audio, X_text = load_rq4_data()
    X_by_channel = {"audio": X_audio, "text": X_text, "both": np.hstack([X_audio, X_text])}

    metrics, preds = [], []
    for split_name in args.splits:
        col = SPLITS[split_name]
        tr = (df[col] == "train").to_numpy()
        va = (df[col] == "val").to_numpy()
        te = (df[col] == "test").to_numpy()
        if tr.sum() == 0 or te.sum() == 0:
            print(f"{split_name}: skipping, empty train/test on the lyrics subset")
            continue
        songs_te = df.loc[te, "song_id"].to_numpy()

        for model in args.models:
            for target in REG_TARGETS:
                y = df[target].to_numpy(dtype=float)
                for channel, X in X_by_channel.items():
                    yhat, info = fit_regression(X[tr], y[tr], X[va], y[va], X[te], model)
                    yte = y[te]
                    lo, hi = boot_ci(r2_score, yte, yhat, n=args.boot)
                    metrics.append({
                        "split": split_name, "model": model, "target": target,
                        "channel": channel, "metric": "r2", "value": r2_score(yte, yhat),
                        "ci_lo": lo, "ci_hi": hi,
                        "rmse": float(np.sqrt(mean_squared_error(yte, yhat))),
                        "spearman": float(spearmanr(yte, yhat).statistic),
                        "n_test": int(te.sum()), **info,
                    })
                    preds.append(pd.DataFrame({
                        "split": split_name, "model": model, "target": target,
                        "channel": channel, "song_id": songs_te,
                        "y_true": yte.astype(float), "y_pred": np.asarray(yhat, dtype=float),
                    }))
                print(f"{split_name:<15} {model:<5} {target:<8} done", flush=True)

    mdf = pd.DataFrame(metrics)
    mdf.to_csv(OUT / "rq4_results.csv", index=False)
    print(f"\nwrote rq4_results.csv rows={len(mdf)}")

    pdf = pd.concat(preds, ignore_index=True)
    pdf.to_parquet(OUT / "rq4_test_predictions.parquet", index=False)
    print(f"wrote rq4_test_predictions.parquet rows={len(pdf)}")

    for name, out in [("rq4_combo_vs_best", combo_vs_best(pdf)),
                      ("rq4_error_correlation", error_correlation(pdf)),
                      ("rq4_divergent_tracks", divergent_tracks(pdf))]:
        out.to_csv(OUT / f"{name}.csv", index=False)
        print(f"wrote {name}.csv rows={len(out)}")

    primary = mdf[(mdf["split"] == "official") & (mdf["model"] == "ridge")]
    if not primary.empty:
        print("\n--- official / ridge, R^2 by channel ---")
        print(primary.pivot_table(index="target", columns="channel", values="value")
                     .round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
