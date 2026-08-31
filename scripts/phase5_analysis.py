"""Phase 5 - hypothesis tests, controls and the RQ1-RQ3 analysis tables.

Everything here reads the persisted Phase 4 test predictions, so no model is refitted.
The same bootstrap resample indices are reused across conditions: every condition is
evaluated on the identical tracks, so an unpaired test would badly overstate the
significance of between-stem differences.

Outputs (all in outputs/):
  analysis_condition_ranking.csv   per-condition scores, ranked, per target
  analysis_paired_vs_mix.csv       paired bootstrap of each condition against `mix`
  analysis_paired_pairs.csv        the specific H1/H2 head-to-head contrasts
  analysis_silent_sensitivity.csv  scores with near-silent conditions excluded (C4)
  analysis_energy_density.csv      predictive power per unit signal energy (H3)
  analysis_controls.csv            remix-vs-mix and residual checks (C2)
  analysis_artifact_null.csv       real vs spectral-shuffled conditions (C3), written
                                   when the shuffle-tagged phase 4 predictions exist
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import CONDITIONS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

N_BOOT = 2000
SEED = 42
SILENT_ENERGY_SHARE = 0.01     # a condition below 1% of mix RMS is treated as absent
STEM_ORDER = list(CONDITIONS)


def load(tag: str = ""):
    suffix = f"_{tag}" if tag else ""
    reg = pd.read_parquet(OUT / f"test_predictions_reg{suffix}.parquet")
    clf_path = OUT / f"test_predictions_clf{suffix}.parquet"
    clf = pd.read_parquet(clf_path) if clf_path.exists() else pd.DataFrame()
    feat = pd.read_parquet(OUT / "stem_features.parquet",
                           columns=["song_id", "condition", "energy_share", "lufs", "silent"])
    return reg, clf, feat


def _wide(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """song_id x condition matrix of `value`, keeping only songs present in every condition."""
    w = df.pivot_table(index="song_id", columns="condition", values=value, aggfunc="first")
    return w.dropna(axis=0, how="any")


def paired_bootstrap(truth: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                     metric, n: int = N_BOOT, seed: int = SEED):
    """Bootstrap the metric DIFFERENCE (a - b) on identical resamples of the same tracks."""
    rng = np.random.default_rng(seed)
    m = len(truth)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        diffs[i] = metric(truth[idx], pred_a[idx]) - metric(truth[idx], pred_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)


def condition_ranking(reg, clf) -> pd.DataFrame:
    rows = []
    for df, metric_fn, metric_name in [(reg, r2_score, "r2"),
                                       (clf, lambda a, b: f1_score(a, b, average="macro"),
                                        "macro_f1")]:
        if df.empty:
            continue
        keys = ["split", "featset", "model", "target"]
        for key, g in df.groupby(keys):
            for cond, gc in g.groupby("condition"):
                rows.append(dict(zip(keys, key)) | {
                    "condition": cond, "metric": metric_name,
                    "value": metric_fn(gc["y_true"].to_numpy(), gc["y_pred"].to_numpy()),
                    "n": len(gc),
                })
    out = pd.DataFrame(rows)
    out["rank"] = out.groupby(["split", "featset", "model", "target"])["value"] \
                     .rank(ascending=False, method="min")
    return out.sort_values(["split", "featset", "model", "target", "rank"])


def paired_vs_reference(reg, clf, reference: str = "mix") -> pd.DataFrame:
    rows = []
    specs = [(reg, r2_score, "r2"), (clf, lambda a, b: f1_score(a, b, average="macro"), "macro_f1")]
    for df, metric_fn, metric_name in specs:
        if df.empty:
            continue
        for key, g in df.groupby(["split", "featset", "model", "target"]):
            truth_w = _wide(g, "y_true")
            pred_w = _wide(g, "y_pred")
            common = truth_w.index.intersection(pred_w.index)
            if reference not in pred_w.columns or len(common) == 0:
                continue
            truth = truth_w.loc[common, reference].to_numpy()
            for cond in pred_w.columns:
                if cond == reference:
                    continue
                d, lo, hi, sig = paired_bootstrap(
                    truth,
                    pred_w.loc[common, cond].to_numpy(),
                    pred_w.loc[common, reference].to_numpy(),
                    metric_fn,
                )
                rows.append({
                    "split": key[0], "featset": key[1], "model": key[2], "target": key[3],
                    "condition": cond, "reference": reference, "metric": metric_name,
                    "diff": d, "ci_lo": lo, "ci_hi": hi, "significant": sig, "n": len(common),
                })
    return pd.DataFrame(rows)


def hypothesis_pairs(reg) -> pd.DataFrame:
    """The specific head-to-head contrasts H1 and H2 are stated in terms of."""
    contrasts = [
        # (condition_a, condition_b, target, hypothesis, expectation)
        ("drums", "other", "arousal", "H1", "drums > other for arousal"),
        ("drums", "other", "valence", "H2", "drums < other for valence"),
        ("bass", "other", "arousal", "H1", "bass > other for arousal"),
        ("vocals", "drums", "valence", "H2", "vocals > drums for valence"),
        ("other", "drums", "valence", "H2", "other > drums for valence"),
        # Sufficiency comparison: voice-removed audio vs voice alone. This is NOT the
        # "voice removal hurts valence" test -- that is accompaniment vs `mix`, which
        # analysis_paired_vs_mix.csv already reports.
        ("accompaniment", "vocals", "valence", "H2", "accompaniment > vocals alone for valence"),
    ]
    rows = []
    for key, g in reg.groupby(["split", "featset", "model", "target"]):
        truth_w, pred_w = _wide(g, "y_true"), _wide(g, "y_pred")
        common = truth_w.index.intersection(pred_w.index)
        if not len(common):
            continue
        for a, b, target, hyp, expect in contrasts:
            if target != key[3] or a not in pred_w.columns or b not in pred_w.columns:
                continue
            truth = truth_w.loc[common].iloc[:, 0].to_numpy()
            d, lo, hi, sig = paired_bootstrap(
                truth, pred_w.loc[common, a].to_numpy(), pred_w.loc[common, b].to_numpy(),
                r2_score,
            )
            rows.append({
                "split": key[0], "featset": key[1], "model": key[2], "target": target,
                "hypothesis": hyp, "contrast": f"{a} - {b}", "expectation": expect,
                "diff_r2": d, "ci_lo": lo, "ci_hi": hi, "significant": sig,
            })
    return pd.DataFrame(rows)


def silent_sensitivity(reg, feat, thresh: float = SILENT_ENERGY_SHARE) -> pd.DataFrame:
    """C4: does a stem's apparent power survive dropping tracks where it is near-absent?"""
    share = feat.set_index(["song_id", "condition"])["energy_share"]
    rows = []
    for key, g in reg.groupby(["split", "featset", "model", "target", "condition"]):
        s = share.reindex(pd.MultiIndex.from_arrays(
            [g["song_id"], g["condition"]])).to_numpy()
        keep = np.isfinite(s) & (s >= thresh)
        full = r2_score(g["y_true"], g["y_pred"])
        kept = r2_score(g["y_true"][keep], g["y_pred"][keep]) if keep.sum() > 30 else np.nan
        rows.append({
            "split": key[0], "featset": key[1], "model": key[2], "target": key[3],
            "condition": key[4], "r2_all": full, "r2_present_only": kept,
            "n_all": len(g), "n_present": int(keep.sum()),
            "frac_absent": float(1 - keep.mean()),
        })
    return pd.DataFrame(rows)


def energy_density(ranking: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """H3: predictive power relative to the share of signal energy the stem occupies."""
    med = feat.groupby("condition")["energy_share"].median().rename("median_energy_share")
    df = ranking.merge(med, left_on="condition", right_index=True, how="left")
    df["value_per_energy"] = df["value"] / df["median_energy_share"].replace(0, np.nan)
    return df.sort_values(["split", "featset", "model", "target", "value_per_energy"],
                          ascending=[True, True, True, True, False])


def _boot_value(truth: np.ndarray, pred: np.ndarray, metric,
                n: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """Plain bootstrap CI of a single condition's score."""
    rng = np.random.default_rng(seed)
    m = len(truth)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        vals[i] = metric(truth[idx], pred[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def artifact_null(reg, clf) -> pd.DataFrame:
    """C3: each condition against its spectral-shuffled null.

    The shuffle preserves the energy envelope (frame-level spectral energy) and destroys
    all fine structure, so `value_shuffle` is what trivial energy information alone
    achieves, and `diff` (real - shuffled, paired) is the part of the score that needed
    actual musical content. A condition whose diff CI includes zero is carrying nothing
    beyond its envelope - its apparent power would be compatible with a separator
    artifact that merely modulates energy.
    """
    rows = []
    specs = [(reg, r2_score, "r2"), (clf, lambda a, b: f1_score(a, b, average="macro"), "macro_f1")]
    for df, metric_fn, metric_name in specs:
        if df.empty:
            continue
        shuffled = sorted(c for c in df["condition"].unique() if c.startswith("shuffle_"))
        if not shuffled:
            continue
        for key, g in df.groupby(["split", "featset", "model", "target"]):
            truth_w, pred_w = _wide(g, "y_true"), _wide(g, "y_pred")
            common = truth_w.index.intersection(pred_w.index)
            for sc in shuffled:
                base = sc[len("shuffle_"):]
                if base not in pred_w.columns or sc not in pred_w.columns or not len(common):
                    continue
                truth = truth_w.loc[common, base].to_numpy()
                real = pred_w.loc[common, base].to_numpy()
                null = pred_w.loc[common, sc].to_numpy()
                d, lo, hi, sig = paired_bootstrap(truth, real, null, metric_fn)
                s_lo, s_hi = _boot_value(truth, null, metric_fn)
                rows.append({
                    "split": key[0], "featset": key[1], "model": key[2], "target": key[3],
                    "condition": base, "metric": metric_name,
                    "value_real": metric_fn(truth, real),
                    "value_shuffle": metric_fn(truth, null),
                    "shuffle_ci_lo": s_lo, "shuffle_ci_hi": s_hi,
                    "diff_real_minus_shuffle": d, "ci_lo": lo, "ci_hi": hi,
                    "significant": sig, "n": len(common),
                })
    return pd.DataFrame(rows)


def controls(reg, clf) -> pd.DataFrame:
    """C2: remix should match mix; residual should sit at baseline."""
    rows = []
    specs = [(reg, r2_score, "r2"), (clf, lambda a, b: f1_score(a, b, average="macro"), "macro_f1")]
    for df, metric_fn, metric_name in specs:
        if df.empty:
            continue
        for key, g in df.groupby(["split", "featset", "model", "target"]):
            vals = {c: metric_fn(gc["y_true"], gc["y_pred"]) for c, gc in g.groupby("condition")}
            rows.append({
                "split": key[0], "featset": key[1], "model": key[2], "target": key[3],
                "metric": metric_name,
                "mix": vals.get("mix"), "remix": vals.get("remix"),
                "remix_minus_mix": (vals.get("remix", np.nan) - vals.get("mix", np.nan)),
                "residual": vals.get("residual"),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    # The primary spec is fixed a priori, NOT chosen by test performance: `both` is the
    # full feature budget every condition receives (C5), and ridge is the plain primary
    # estimator. Every featset/model cell is still written to the CSVs regardless.
    ap.add_argument("--primary-featset", default="both")
    ap.add_argument("--primary-model", default="ridge")
    args = ap.parse_args()

    reg, clf, feat = load(args.tag)
    print(f"regression preds={len(reg)}  quadrant preds={len(clf)}  feature rows={len(feat)}")

    suffix = f"_{args.tag}" if args.tag else ""
    ranking = condition_ranking(reg, clf)
    outputs = {
        "analysis_condition_ranking": ranking,
        "analysis_paired_vs_mix": paired_vs_reference(reg, clf),
        "analysis_paired_pairs": hypothesis_pairs(reg),
        "analysis_silent_sensitivity": silent_sensitivity(reg, feat),
        "analysis_energy_density": energy_density(ranking, feat),
        "analysis_controls": controls(reg, clf),
    }

    # C3: the shuffle-null predictions live under their own tag (they come from a
    # separate feature table). Fold them in for the artifact-null table only, so the
    # main tables keep their canonical 11 conditions.
    shuf_reg_path = OUT / "test_predictions_reg_shuffle.parquet"
    shuf_clf_path = OUT / "test_predictions_clf_shuffle.parquet"
    if not args.tag and shuf_reg_path.exists():
        reg_n = pd.concat([reg, pd.read_parquet(shuf_reg_path)], ignore_index=True)
        clf_n = pd.concat([clf, pd.read_parquet(shuf_clf_path)], ignore_index=True) \
            if shuf_clf_path.exists() else clf
        outputs["analysis_artifact_null"] = artifact_null(reg_n, clf_n)
    elif not args.tag:
        print("C3 shuffle predictions not found - skipping analysis_artifact_null "
              "(run phase4 with --features stem_features_shuffle.parquet --tag shuffle)")

    for name, df in outputs.items():
        path = OUT / f"{name}{suffix}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path.name} rows={len(df)}")

    # headline view: primary spec only
    primary = ranking[(ranking["featset"] == args.primary_featset)
                      & (ranking["model"] == args.primary_model)]
    if primary.empty:
        print(f"\nprimary spec {args.primary_featset}/{args.primary_model} absent; "
              f"showing all cells")
        primary = ranking
    for target in ["arousal", "valence", "quadrant"]:
        sub = primary[primary["target"] == target]
        if sub.empty:
            continue
        print(f"\n--- {target} ({args.primary_featset}/{args.primary_model}) ---")
        piv = sub.pivot_table(index="condition", columns="split", values="value")
        print(piv.reindex([c for c in STEM_ORDER if c in piv.index]).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
