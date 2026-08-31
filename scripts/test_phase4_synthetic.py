"""Validate the Phase 4 pipeline on synthetic data with a KNOWN answer.

Builds fake features where the signal is planted in specific conditions:
  * `drums`  carries arousal only
  * `other`  carries valence only
  * `vocals` carries both, weakly
  * `bass`   carries nothing
If Phase 4 is wired correctly it must recover exactly that pattern. This catches
index-misalignment and target-leakage bugs before the real 8 h feature run.

Run:  python scripts/test_phase4_synthetic.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
TMP = OUT / "_synthetic_check"

N = 600
DIM = 32
SEED = 0
PLANTED = {
    "drums": ("arousal", 1.0),
    "other": ("valence", 1.0),
    "vocals": ("both", 0.5),
    "bass": (None, 0.0),
}
CONDS = ["mix", "vocals", "drums", "bass", "other", "accompaniment",
         "no_drums", "no_bass", "no_other", "remix", "residual"]


def main() -> int:
    rng = np.random.default_rng(SEED)
    TMP.mkdir(parents=True, exist_ok=True)

    arousal = rng.uniform(0, 1, N)
    valence = rng.uniform(0, 1, N)
    quadrant = np.where(arousal > 0.5,
                        np.where(valence > 0.5, "Q1", "Q2"),
                        np.where(valence > 0.5, "Q4", "Q3"))
    song_id = np.array([f"S{i:04d}" for i in range(N)])
    split = np.array(["train"] * int(N * 0.7) + ["val"] * int(N * 0.15)
                     + ["test"] * (N - int(N * 0.7) - int(N * 0.15)))
    rng.shuffle(split)

    man = pd.DataFrame({
        "song_id": song_id, "arousal": arousal, "valence": valence, "quadrant": quadrant,
        "split_70_15_15": split, "split_artist_disjoint": split,
    })

    rows = []
    for cond in CONDS:
        target, strength = PLANTED.get(cond, (None, 0.0))
        X = rng.normal(0, 1, (N, DIM)).astype(np.float32)
        if target in ("arousal", "both"):
            X[:, 0] += strength * 4 * arousal
        if target in ("valence", "both"):
            X[:, 1] += strength * 4 * valence
        for i in range(N):
            rows.append({"song_id": song_id[i], "condition": cond,
                         "handcrafted": X[i], "silent": False,
                         "energy_share": 1.0, "error": None})

    (TMP / "outputs").mkdir(parents=True, exist_ok=True)
    man.to_parquet(TMP / "manifest.parquet", index=False)
    pd.DataFrame(rows).to_parquet(TMP / "stem_features.parquet", index=False)

    # run phase4 against the synthetic outputs dir
    real_man, real_feat = OUT / "manifest.parquet", OUT / "stem_features.parquet"
    bak_man, bak_feat = OUT / "manifest.real.bak", OUT / "stem_features.real.bak"
    for src, bak in [(real_man, bak_man), (real_feat, bak_feat)]:
        if src.exists():
            shutil.move(src, bak)
    try:
        shutil.copy(TMP / "manifest.parquet", real_man)
        shutil.copy(TMP / "stem_features.parquet", real_feat)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "phase4_models.py"),
             "--featsets", "handcrafted", "--splits", "official",
             "--boot", "200", "--tag", "synthetic"],
            check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
        )
    finally:
        for src, bak in [(real_man, bak_man), (real_feat, bak_feat)]:
            src.unlink(missing_ok=True)
            if bak.exists():
                shutil.move(bak, src)

    res = pd.read_csv(OUT / "results_metrics_synthetic.csv")
    piv = res.pivot_table(index="condition", columns="target", values="value")
    piv = piv.reindex(CONDS)
    print("\nRecovered signal (R2 for arousal/valence, macro-F1 for quadrant):")
    print(piv.round(3).to_string())

    # Rank/margin based: absolute R2 depends on the noise draw, but the ORDERING and the
    # separation from the unplanted conditions are what a correct pipeline must reproduce.
    null_conds = [c for c in CONDS if c not in PLANTED]
    null_max = {t: piv.loc[null_conds, t].max() for t in ("arousal", "valence")}
    MARGIN = 0.15

    ok = True
    checks = [
        ("arousal is recovered best from drums",
         piv["arousal"].idxmax() == "drums"
         and piv.loc["drums", "arousal"] > null_max["arousal"] + MARGIN),
        ("valence is recovered best from other",
         piv["valence"].idxmax() == "other"
         and piv.loc["other", "valence"] > null_max["valence"] + MARGIN),
        ("drums carries no valence, other carries no arousal",
         piv.loc["drums", "valence"] < null_max["valence"] + MARGIN
         and piv.loc["other", "arousal"] < null_max["arousal"] + MARGIN),
        ("vocals carries both, and less than the dedicated stems",
         piv.loc["vocals", "arousal"] > null_max["arousal"] + 0.05
         and piv.loc["vocals", "valence"] > null_max["valence"] + 0.05
         and piv.loc["vocals", "arousal"] < piv.loc["drums", "arousal"]
         and piv.loc["vocals", "valence"] < piv.loc["other", "valence"]),
        ("bass carries nothing",
         piv.loc["bass", "arousal"] < null_max["arousal"] + MARGIN
         and piv.loc["bass", "valence"] < null_max["valence"] + MARGIN),
    ]
    print()
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= bool(passed)

    for f in ["results_metrics_synthetic.csv", "test_predictions_reg_synthetic.parquet",
              "test_predictions_clf_synthetic.parquet"]:
        (OUT / f).unlink(missing_ok=True)
    shutil.rmtree(TMP, ignore_errors=True)

    print("\nSYNTHETIC CHECK:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
