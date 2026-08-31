"""Phase 4 - per-condition prediction of arousal, valence and quadrant.

The contribution is the comparison ACROSS conditions, not the model, so the primary
estimator is deliberately plain: ridge / logistic regression on standardised features,
with the regularisation strength chosen on the validation split only. A gradient-boosted
secondary model is run as a robustness check to confirm the ranking is not an artifact of
linearity.

Test predictions are persisted per track so Phase 5 can do error-correlation analysis
(RQ4) and the paired bootstrap without refitting anything.

Outputs
  outputs/results_metrics.csv       one row per (split, featset, model, condition, target)
  outputs/test_predictions.parquet  per-track test predictions for every cell
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import CONDITIONS  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

SPLITS = {"official": "split_70_15_15", "artist_disjoint": "split_artist_disjoint"}
FEATSETS = ["handcrafted", "clap", "both"]
REG_TARGETS = ["arousal", "valence"]
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
LOGREG_CS = [0.01, 0.1, 1.0, 10.0]
SEED = 42


def build_matrix(feat: pd.DataFrame, condition: str, featset: str) -> tuple[pd.Index, np.ndarray]:
    sub = feat[feat["condition"] == condition]
    blocks, ok = [], np.ones(len(sub), dtype=bool)
    for col in (["handcrafted", "clap"] if featset == "both" else [featset]):
        vals = sub[col].to_numpy()
        present = np.array([v is not None and np.ndim(v) == 1 for v in vals])
        ok &= present
        dim = next((len(v) for v in vals if v is not None and np.ndim(v) == 1), 0)
        blocks.append(np.stack([
            np.asarray(v, dtype=np.float32) if (v is not None and np.ndim(v) == 1)
            else np.zeros(dim, np.float32) for v in vals
        ]))
    X = np.hstack(blocks)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return sub.index[ok], X[ok]


def boot_ci(fn, *arrays, n: int = 1000, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        try:
            vals.append(fn(*[a[idx] for a in arrays]))
        except Exception:
            continue
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def fit_regression(Xtr, ytr, Xva, yva, Xte, model: str):
    if model == "ridge":
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
        best, best_score, best_a = None, -np.inf, RIDGE_ALPHAS[0]
        for a in RIDGE_ALPHAS:
            est = Ridge(alpha=a).fit(Xtr_s, ytr)
            s = r2_score(yva, est.predict(Xva_s))
            if s > best_score:
                best, best_score, best_a = est, s, a
        # a degenerate condition (e.g. an all-zero stem) can score NaN for every alpha;
        # fall back to the first alpha rather than crashing the whole sweep
        if best is None:
            best = Ridge(alpha=best_a).fit(Xtr_s, ytr)
        return best.predict(Xte_s), {"alpha": best_a, "val_r2": best_score}
    est = HistGradientBoostingRegressor(random_state=SEED, max_iter=300,
                                        early_stopping=True, validation_fraction=0.15)
    est.fit(Xtr, ytr)
    return est.predict(Xte), {"val_r2": r2_score(yva, est.predict(Xva))}


def fit_classification(Xtr, ytr, Xva, yva, Xte, model: str):
    if model == "ridge":
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
        best, best_score, best_c = None, -np.inf, None
        for c in LOGREG_CS:
            est = LogisticRegression(C=c, max_iter=2000).fit(Xtr_s, ytr)
            s = f1_score(yva, est.predict(Xva_s), average="macro")
            if s > best_score:
                best, best_score, best_c = est, s, c
        return best.predict(Xte_s), {"C": best_c, "val_macro_f1": best_score}
    est = HistGradientBoostingClassifier(random_state=SEED, max_iter=300,
                                         early_stopping=True, validation_fraction=0.15)
    est.fit(Xtr, ytr)
    return est.predict(Xte), {"val_macro_f1": f1_score(yva, est.predict(Xva), average="macro")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ridge"], choices=["ridge", "gbm"])
    ap.add_argument("--featsets", nargs="+", default=FEATSETS)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--features", default="stem_features.parquet",
                    help="feature table in outputs/ (stem_features_raw.parquet for the C1 "
                         "un-normalised pass, stem_features_shuffle.parquet for the C3 null)")
    args = ap.parse_args()

    man = pd.read_parquet(OUT / "manifest.parquet")
    feat = pd.read_parquet(OUT / args.features)
    feat = feat.merge(
        man[["song_id", "quadrant", "arousal", "valence", *SPLITS.values()]],
        on="song_id", how="left",
    )
    present = set(feat["condition"])
    # canonical conditions in their declared order, then any extras (e.g. the C3
    # shuffle_* nulls) in a stable alphabetical tail
    conditions = [c for c in CONDITIONS if c in present] \
        + sorted(c for c in present if c not in CONDITIONS)
    available = {f for f in ("handcrafted", "clap") if f in feat.columns}
    needed = {"handcrafted": {"handcrafted"}, "clap": {"clap"},
              "both": {"handcrafted", "clap"}}
    featsets = [f for f in args.featsets if needed[f] <= available]
    if set(featsets) != set(args.featsets):
        print(f"stem_features.parquet has {sorted(available)}; using featsets {featsets}")
    print(f"features rows={len(feat)} songs={feat['song_id'].nunique()} conditions={conditions}")

    metrics, preds_reg, preds_clf = [], [], []
    t0 = time.perf_counter()
    for split_name in args.splits:
        col = SPLITS[split_name]
        for featset in featsets:
            for model in args.models:
                for cond in conditions:
                    idx, X = build_matrix(feat, cond, featset)
                    sub = feat.loc[idx]
                    tr = (sub[col] == "train").to_numpy()
                    va = (sub[col] == "val").to_numpy()
                    te = (sub[col] == "test").to_numpy()
                    if tr.sum() == 0 or te.sum() == 0:
                        continue
                    Xtr, Xva, Xte = X[tr], X[va], X[te]
                    songs_te = sub.loc[te, "song_id"].to_numpy()

                    for target in REG_TARGETS:
                        y = sub[target].to_numpy(dtype=float)
                        yhat, info = fit_regression(Xtr, y[tr], Xva, y[va], Xte, model)
                        yte = y[te]
                        r2 = r2_score(yte, yhat)
                        rmse = float(np.sqrt(mean_squared_error(yte, yhat)))
                        rho = float(spearmanr(yte, yhat).statistic)
                        lo, hi = boot_ci(lambda a, b: r2_score(a, b), yte, yhat, n=args.boot)
                        # predict-the-mean baseline, fit on train
                        base = r2_score(yte, np.full_like(yte, y[tr].mean()))
                        metrics.append({
                            "split": split_name, "featset": featset, "model": model,
                            "condition": cond, "target": target, "metric": "r2",
                            "value": r2, "ci_lo": lo, "ci_hi": hi, "rmse": rmse,
                            "spearman": rho, "baseline": base, "n_test": int(te.sum()),
                            **info,
                        })
                        preds_reg.append(pd.DataFrame({
                            "split": split_name, "featset": featset, "model": model,
                            "condition": cond, "target": target, "song_id": songs_te,
                            "y_true": yte.astype(float), "y_pred": np.asarray(yhat, dtype=float),
                        }))

                    yq = sub["quadrant"].to_numpy()
                    qhat, info = fit_classification(Xtr, yq[tr], Xva, yq[va], Xte, model)
                    qte = yq[te]
                    acc = accuracy_score(qte, qhat)
                    mf1 = f1_score(qte, qhat, average="macro")
                    lo, hi = boot_ci(lambda a, b: f1_score(a, b, average="macro"),
                                     qte, qhat, n=args.boot)
                    metrics.append({
                        "split": split_name, "featset": featset, "model": model,
                        "condition": cond, "target": "quadrant", "metric": "macro_f1",
                        "value": mf1, "ci_lo": lo, "ci_hi": hi, "accuracy": acc,
                        "baseline": 0.25, "n_test": int(te.sum()), **info,
                    })
                    preds_clf.append(pd.DataFrame({
                        "split": split_name, "featset": featset, "model": model,
                        "condition": cond, "target": "quadrant", "song_id": songs_te,
                        "y_true": qte.astype(str), "y_pred": np.asarray(qhat, dtype=str),
                    }))

                    el = time.perf_counter() - t0
                    print(f"[{el/60:6.1f}m] {split_name:<15} {featset:<11} {model:<5} "
                          f"{cond:<14} done", flush=True)

    if not metrics:
        print("no fittable cells - every condition lacked a populated train or test split")
        return 1

    mdf = pd.DataFrame(metrics)
    suffix = f"_{args.tag}" if args.tag else ""
    mdf.to_csv(OUT / f"results_metrics{suffix}.csv", index=False)
    print(f"\nwrote results_metrics{suffix}.csv rows={len(mdf)}")

    # regression and quadrant predictions are stored apart: one has float labels,
    # the other string labels, and parquet will not hold both in one column
    for name, chunks in [("reg", preds_reg), ("clf", preds_clf)]:
        if not chunks:
            continue
        pdf = pd.concat(chunks, ignore_index=True)
        path = OUT / f"test_predictions_{name}{suffix}.parquet"
        pdf.to_parquet(path, index=False)
        print(f"wrote {path.name} rows={len(pdf)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
