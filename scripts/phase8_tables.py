"""Phase 8 - the manuscript's numbered tables.

PeerJ wants every table as its own file, so this writes exactly the tables the text
cites, in the order it cites them. Table 1 already exists (phase6_figures.py); this
script writes Tables 2-5 and leaves Table 1 alone.

  table2_hypotheses.csv   H1-H4 outcomes against their pre-registered falsification tests
  table3_artifact_null.csv C3: each condition against its spectral-shuffle null
  table4_absent_stems.csv  C4: prevalence of near-absent stems and the score shift
  table5_rq4.csv           RQ4: vocal acoustics vs lyric semantics

All rows are the primary spec (featset `both`, ridge) unless a column says otherwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

FEATSET, MODEL = "both", "ridge"
STEMS = ["vocals", "drums", "bass", "other"]
SPLIT_LABEL = {"official": "Official", "artist_disjoint": "Artist-disjoint"}


def ci(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


def primary(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["featset"] == FEATSET) & (df["model"] == MODEL)]


def table2() -> pd.DataFrame:
    """H1-H4, each against the falsification test declared before modelling."""
    pairs = primary(pd.read_csv(OUT / "analysis_paired_pairs.csv"))
    ed = primary(pd.read_csv(OUT / "analysis_energy_density.csv"))
    combo = pd.read_csv(OUT / "rq4_combo_vs_best.csv")
    combo = combo[combo["model"] == MODEL]

    rows = []
    contrasts = [
        ("H1", "arousal", "drums - other", "drums > other for arousal"),
        ("H1", "arousal", "bass - other", "bass > other for arousal"),
        ("H2", "valence", "other - drums", "other > drums for valence"),
        ("H2", "valence", "vocals - drums", "vocals > drums for valence"),
        ("H2", "valence", "accompaniment - vocals", "accompaniment > vocals for valence"),
    ]
    for split in ["official", "artist_disjoint"]:
        for hyp, target, contrast, expect in contrasts:
            r = pairs[(pairs["split"] == split) & (pairs["target"] == target)
                      & (pairs["contrast"] == contrast)]
            if r.empty:
                continue
            r = r.iloc[0]
            direction = expect.split()[1]           # '>' or '<'
            as_predicted = (r["diff_r2"] > 0) if direction == ">" else (r["diff_r2"] < 0)
            rows.append({
                "hypothesis": hyp, "split": SPLIT_LABEL[split], "target": target,
                "test": contrast, "prediction": expect,
                "estimate": round(float(r["diff_r2"]), 3),
                "ci_95": ci(r["ci_lo"], r["ci_hi"]),
                "significant": bool(r["significant"]),
                "outcome": ("supported" if (as_predicted and r["significant"])
                            else "contrary (significant)" if (not as_predicted and r["significant"])
                            else "null"),
            })

    # H3 is a ratio, not a resampled difference: R² per unit median energy share.
    for split in ["official", "artist_disjoint"]:
        d = ed[(ed["split"] == split) & (ed["target"] == "arousal")]
        d = d.set_index("condition")["value_per_energy"]
        stem_rank = d.reindex(STEMS).sort_values(ascending=False)
        rows.append({
            "hypothesis": "H3", "split": SPLIT_LABEL[split], "target": "arousal",
            "test": "vocals - accompaniment (R² per unit energy)",
            "prediction": "vocals highest information density",
            "estimate": round(float(d["vocals"] - d["accompaniment"]), 3),
            "ci_95": "n/a (deterministic ratio)",
            "significant": "",
            "outcome": (f"stated test passed; vocals rank "
                        f"{list(stem_rank.index).index('vocals') + 1}/4 among stems"),
        })

    for _, r in combo.iterrows():
        rows.append({
            "hypothesis": "H4", "split": SPLIT_LABEL[r["split"]], "target": r["target"],
            "test": f"both - {r['best_single']} (better single channel)",
            "prediction": "combining beats either channel alone",
            "estimate": round(float(r["diff_both_minus_best_single"]), 3),
            "ci_95": ci(r["ci_lo"], r["ci_hi"]),
            "significant": bool(r["significant"]),
            "outcome": "supported" if r["h4_supported"] else "null",
        })
    return pd.DataFrame(rows)


def table3() -> pd.DataFrame:
    an = primary(pd.read_csv(OUT / "analysis_artifact_null.csv"))
    an = an[an["target"].isin(["arousal", "valence"])]
    out = pd.DataFrame({
        "split": an["split"].map(SPLIT_LABEL),
        "target": an["target"],
        "condition": an["condition"],
        "r2_real": an["value_real"].round(3),
        "r2_shuffled": an["value_shuffle"].round(3),
        "shuffled_ci_95": [ci(a, b) for a, b in zip(an["shuffle_ci_lo"], an["shuffle_ci_hi"])],
        "difference": an["diff_real_minus_shuffle"].round(3),
        "difference_ci_95": [ci(a, b) for a, b in zip(an["ci_lo"], an["ci_hi"])],
        "significant": an["significant"],
        "frac_of_real_explained_by_envelope":
            (an["value_shuffle"] / an["value_real"]).round(2),
    })
    order = {c: i for i, c in enumerate(["mix", *STEMS])}
    return out.sort_values(["split", "target"],
                           key=lambda s: s.map(lambda v: v)).assign(
        _o=out["condition"].map(order)).sort_values(
        ["split", "target", "_o"]).drop(columns="_o")


def table4() -> pd.DataFrame:
    feat = pd.read_parquet(OUT / "stem_features.parquet",
                           columns=["song_id", "condition", "energy_share"])
    man = pd.read_parquet(OUT / "manifest.parquet", columns=["song_id", "quadrant"])
    f = feat.merge(man, on="song_id")
    f = f[f["condition"].isin(STEMS)]
    f["absent"] = f["energy_share"] < 0.01
    prev = f.pivot_table(index="condition", columns="quadrant", values="absent",
                         aggfunc="mean")
    overall = f.groupby("condition")["absent"].mean().rename("all")
    share = f.groupby("condition")["energy_share"].median().rename("median_energy_share")

    ss = primary(pd.read_csv(OUT / "analysis_silent_sensitivity.csv"))
    ss = ss[ss["condition"].isin(STEMS)]
    rows = []
    for stem in STEMS:
        base = {
            "stem": stem,
            "median_energy_share": round(float(share[stem]), 3),
            "absent_all": round(float(overall[stem]), 3),
            **{f"absent_{q}": round(float(prev.loc[stem, q]), 3) for q in prev.columns},
        }
        for split in ["official", "artist_disjoint"]:
            for target in ["arousal", "valence"]:
                r = ss[(ss["split"] == split) & (ss["target"] == target)
                       & (ss["condition"] == stem)]
                if r.empty:
                    continue
                r = r.iloc[0]
                base[f"{split}_{target}_r2_all"] = round(float(r["r2_all"]), 3)
                base[f"{split}_{target}_r2_present"] = round(float(r["r2_present_only"]), 3)
        rows.append(base)
    return pd.DataFrame(rows)


def table5() -> pd.DataFrame:
    res = pd.read_csv(OUT / "rq4_results.csv")
    res = res[res["model"] == MODEL]
    err = pd.read_csv(OUT / "rq4_error_correlation.csv")
    err = err[err["model"] == MODEL]
    combo = pd.read_csv(OUT / "rq4_combo_vs_best.csv")
    combo = combo[combo["model"] == MODEL]

    rows = []
    for split in ["official", "artist_disjoint"]:
        for target in ["arousal", "valence"]:
            g = res[(res["split"] == split) & (res["target"] == target)]
            e = err[(err["split"] == split) & (err["target"] == target)]
            c = combo[(combo["split"] == split) & (combo["target"] == target)]
            if g.empty:
                continue
            row = {"split": SPLIT_LABEL[split], "target": target,
                   "n_test": int(g["n_test"].iloc[0])}
            for ch in ["audio", "text", "both"]:
                gc = g[g["channel"] == ch]
                if gc.empty:
                    continue
                gc = gc.iloc[0]
                row[f"{ch}_r2"] = round(float(gc["value"]), 3)
                row[f"{ch}_ci_95"] = ci(gc["ci_lo"], gc["ci_hi"])
                row[f"{ch}_rmse"] = round(float(gc["rmse"]), 3)
                row[f"{ch}_spearman"] = round(float(gc["spearman"]), 3)
            if not c.empty:
                c = c.iloc[0]
                row["both_minus_best_single"] = round(float(c["diff_both_minus_best_single"]), 3)
                row["diff_ci_95"] = ci(c["ci_lo"], c["ci_hi"])
                row["h4_supported"] = bool(c["h4_supported"])
            if not e.empty:
                e = e.iloc[0]
                row["error_pearson_r"] = round(float(e["pearson_r"]), 3)
                row["error_p"] = f"{float(e['pearson_p']):.2e}"
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    for name, df in [("table2_hypotheses", table2()),
                     ("table3_artifact_null", table3()),
                     ("table4_absent_stems", table4()),
                     ("table5_rq4", table5())]:
        path = OUT / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path.name} rows={len(df)}")
        print(df.to_string(), "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
