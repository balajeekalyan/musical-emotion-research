"""Phase 6 - manuscript figures and the main results table.

Reads the persisted Phase 4 metrics and Phase 5 analysis tables; fits nothing and
recomputes no bootstrap. Every figure is therefore a pure function of files that are
already on disk, so a figure can never disagree with the numbers in the tables.

Primary spec is the OFFICIAL MERGE split (chosen for comparability with other work on
this dataset), with the `both` feature budget and the plain ridge estimator. The official
split is NOT leakage-free - 341/484 (70%) of its test tracks share an artist with
train/val, and a few conclusions are split-dependent (the sole significant leave-one-out
cell, and H2's other>drums contrast) - so every figure is also emitted for the
artist-disjoint split under its own filename, and the manuscript must present the two
side by side rather than treat the artist-disjoint results as a footnote.

Outputs (outputs/figures/ and outputs/):
  fig1_localization_<split>.{pdf,png}          RQ1/RQ2 headline: where A and V live
  fig2_sufficiency_necessity_<split>.{pdf,png} RQ3: paired diff vs mix, both directions
  fig3_controls_<split>.{pdf,png}              C2: mix / remix / residual
  fig4_energy_density_<split>.{pdf,png}        H3: predictive power per unit energy
  fig5_silent_sensitivity_<split>.{pdf,png}    C4: does power survive absent stems
  fig6_rq4_voice_vs_words_<split>.{pdf,png}    RQ4/H4: vocal audio vs lyric text
  table1_main_results.csv                      the per-condition table for the paper
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIGDIR = OUT / "figures"

# --- design tokens (light print surface; see dataviz reference palette) -------------
SURFACE = "#ffffff"   # PeerJ requires white chart backgrounds
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
S_AROUSAL = "#2a78d6"   # categorical slot 1
S_VALENCE = "#eb6834"   # categorical slot 2
S_THIRD = "#1baf7a"     # categorical slot 3 (direct-labelled; sub-3:1 on this surface)

SINGLE_STEMS = ["vocals", "drums", "bass", "other"]
LEAVE_ONE_OUT = ["accompaniment", "no_drums", "no_bass", "no_other"]
CONTROLS = ["remix", "residual"]
ALL_CONDS = ["mix"] + SINGLE_STEMS + LEAVE_ONE_OUT + CONTROLS

PRETTY = {
    "mix": "mix", "vocals": "vocals", "drums": "drums", "bass": "bass", "other": "other",
    "accompaniment": "accompaniment\n(no vocals)", "no_drums": "no drums",
    "no_bass": "no bass", "no_other": "no other", "remix": "remix", "residual": "residual",
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 150,
    })


LETTER_W_IN, LETTER_H_IN = 8.5, 11.0
MARGIN_IN = 2.5 / 2.54          # PeerJ asks for 2.5 cm on every edge


def save(fig, name: str) -> None:
    """Write the tight figure, plus the US Letter page PeerJ wants for vector uploads.

    The letter page is produced by handing savefig an explicit Bbox rather than by
    post-processing the tight PDF: transforming and re-boxing an existing page through
    PyPDF2 rebuilds its content stream against the reader, which then serialises through
    the writer as an empty page -- resources intact, nothing drawn.
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")

    # tight extent in figure inches, then a letter-sized window around it:
    # centred horizontally, top edge one margin below the top of the page
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    x0 = tight.x0 - (LETTER_W_IN - tight.width) / 2
    y1 = tight.y1 + MARGIN_IN
    page = Bbox([[x0, y1 - LETTER_H_IN], [x0 + LETTER_W_IN, y1]])
    fig.savefig(FIGDIR / f"{name}_letter.pdf", bbox_inches=page)

    plt.close(fig)
    print(f"  wrote figures/{name}.pdf + .png + _letter.pdf")


def err(df: pd.DataFrame) -> np.ndarray:
    """Asymmetric yerr from absolute percentile bounds, clipped so bars never invert."""
    lo = np.clip(df["value"].to_numpy() - df["ci_lo"].to_numpy(), 0, None)
    hi = np.clip(df["ci_hi"].to_numpy() - df["value"].to_numpy(), 0, None)
    return np.vstack([lo, hi])


# --- figure 1: the localization headline -------------------------------------------
def fig1(met: pd.DataFrame, split: str) -> None:
    fig, ax = plt.subplots(figsize=(6.1, 3.9))
    x = np.arange(len(SINGLE_STEMS))
    w = 0.34
    top = 0.0

    for i, (target, color) in enumerate([("arousal", S_AROUSAL), ("valence", S_VALENCE)]):
        d = met[met.target == target].set_index("condition").reindex(SINGLE_STEMS).reset_index()
        off = (i - 0.5) * w
        e = err(d)
        ax.bar(x + off, d["value"], w, color=color, label=target,
               edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.errorbar(x + off, d["value"], yerr=e, fmt="none",
                    ecolor=INK2, elinewidth=0.9, capsize=2.5, zorder=4)
        # label above the CI whisker, never on top of it
        # a surface-coloured backing keeps the value legible where it crosses a
        # dashed mix line
        for xi, v, hi in zip(x + off, d["value"], d["value"] + e[1]):
            ax.text(xi, hi + 0.008, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2, zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
        top = max(top, float((d["value"] + e[1]).max()))

        # the full mix as the ceiling each stem is measured against
        mv = float(met[(met.target == target) & (met.condition == "mix")]["value"].iloc[0])
        ax.axhline(mv, color=color, lw=1.0, ls=(0, (4, 3)), alpha=0.75, zorder=2)
        # label sits outside the plot, where no bar or value label can reach it
        ax.text(1.008, mv, f"mix {mv:.2f}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7.5, color=color, zorder=5)
        top = max(top, mv)

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[c] for c in SINGLE_STEMS])
    ax.set_xlim(-0.55, len(SINGLE_STEMS) - 0.45)
    ax.set_ylim(0, top * 1.16)
    ax.set_ylabel("test $R^2$  (95% CI)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    # legend above the plot so it cannot sit on the mix reference lines
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=8.5)
    # No title or caption in the image: PeerJ carries both in the manuscript caption list.
    save(fig, f"fig1_localization_{split}")


# --- figure 2: sufficiency vs necessity --------------------------------------------
def fig2(pv: pd.DataFrame, split: str) -> None:
    """Forest plot of each condition's paired difference from `mix`.

    Filled marker = the 95% paired-bootstrap CI excludes zero. The RQ3 story is the
    contrast between the two blocks, so they are drawn in one axis with a divider.
    """
    conds = SINGLE_STEMS + LEAVE_ONE_OUT + CONTROLS
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 4.4), sharey=True)
    blocks = [(len(SINGLE_STEMS), "single stems\n(sufficiency)"),
              (len(LEAVE_ONE_OUT), "leave-one-out\n(necessity)"),
              (len(CONTROLS), "controls")]

    for ax, (target, color) in zip(axes, [("arousal", S_AROUSAL), ("valence", S_VALENCE)]):
        d = pv[pv.target == target].set_index("condition").reindex(conds).reset_index()
        y = np.arange(len(conds))[::-1]
        sig = d["significant"].astype(bool).to_numpy()

        ax.axvline(0, color=BASELINE, lw=1.0, zorder=2)
        ax.hlines(y, d["ci_lo"], d["ci_hi"], color=color, lw=1.6, alpha=0.85, zorder=3)
        ax.scatter(d["diff"][sig], y[sig], s=34, color=color,
                   edgecolor=SURFACE, linewidth=1.0, zorder=4)
        ax.scatter(d["diff"][~sig], y[~sig], s=34, facecolor=SURFACE,
                   edgecolor=color, linewidth=1.4, zorder=4)

        start = 0
        for n, _ in blocks[:-1]:
            start += n
            ax.axhline(y[start] + 0.5, color=GRID, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([PRETTY[c].replace("\n", " ") for c in conds])
        ax.set_xlabel(r"$\Delta R^2$ vs mix")
        ax.set_title(target, loc="left", fontsize=10, color=INK)
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.margins(x=0.10)

    # block names outside the right panel, so they can never collide with the data
    start = 0
    for n, label in blocks:
        centre = len(conds) - 1 - (start + (n - 1) / 2)
        axes[1].text(1.035, centre, label, transform=axes[1].get_yaxis_transform(),
                     ha="left", va="center", fontsize=7.5, color=MUTED, linespacing=1.4)
        start += n

    handles = [
        Line2D([], [], marker="o", ls="none", color=INK2, markersize=6,
               label="CI excludes 0 (differs from mix)"),
        Line2D([], [], marker="o", ls="none", markerfacecolor=SURFACE,
               markeredgecolor=INK2, markeredgewidth=1.4, markersize=6,
               label="CI includes 0 (matches mix)"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, -0.02),
               ncol=2, fontsize=8)

    # No title or caption in the image (PeerJ); the manuscript caption carries both.
    fig.tight_layout(rect=(0, 0.02, 0.88, 1.0))
    save(fig, f"fig2_sufficiency_necessity_{split}")


# --- figure 3: reconstruction controls ---------------------------------------------
def fig3(met: pd.DataFrame, split: str) -> None:
    conds = ["mix", "remix", "residual"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = np.arange(len(conds))
    w = 0.34
    top = 0.0
    for i, (target, color) in enumerate([("arousal", S_AROUSAL), ("valence", S_VALENCE)]):
        d = met[met.target == target].set_index("condition").reindex(conds).reset_index()
        off = (i - 0.5) * w
        e = err(d)
        ax.bar(x + off, d["value"], w, color=color, label=target,
               edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.errorbar(x + off, d["value"], yerr=e, fmt="none",
                    ecolor=INK2, elinewidth=0.9, capsize=2.5, zorder=4)
        for xi, v, hi in zip(x + off, d["value"], d["value"] + e[1]):
            ax.text(xi, hi + 0.008, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2, zorder=5)
        top = max(top, float((d["value"] + e[1]).max()))
    ax.axhline(0, color=BASELINE, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylim(0, top * 1.18)
    ax.set_ylabel("test $R^2$  (95% CI)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=8.5)
    save(fig, f"fig3_controls_{split}")


# --- figure 4: energy-normalised density (H3) --------------------------------------
def fig4(ed: pd.DataFrame, split: str) -> None:
    # H3's stated falsification test is vocals against the FULL accompaniment, so the
    # accompaniment has to be on the plot or the figure cannot show the test it settles.
    # It is a composite of three stems, not a fifth stem, so it sits past a divider.
    conds = SINGLE_STEMS + ["accompaniment"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(conds))
    w = 0.34
    top = 0.0
    for i, (target, color) in enumerate([("arousal", S_AROUSAL), ("valence", S_VALENCE)]):
        d = ed[ed.target == target].set_index("condition").reindex(conds).reset_index()
        off = (i - 0.5) * w
        ax.bar(x + off, d["value_per_energy"], w, color=color, label=target,
               edgecolor=SURFACE, linewidth=1.2, zorder=3)
        for xi, v in zip(x + off, d["value_per_energy"]):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2, zorder=5)
        top = max(top, float(d["value_per_energy"].max()))
    ax.axvline(len(SINGLE_STEMS) - 0.5, color=BASELINE, lw=0.9, ls=(0, (3, 3)), zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[c] for c in conds])
    ax.set_ylim(0, top * 1.20)
    ax.text(len(SINGLE_STEMS), top * 1.16, "H3 reference", ha="center", va="top",
            fontsize=7.5, color=MUTED, zorder=5)
    ax.set_ylabel(r"$R^2$ per unit energy share")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=8.5)
    save(fig, f"fig4_energy_density_{split}")


# --- figure 5: silent-stem sensitivity (C4) ----------------------------------------
def fig5(ss: pd.DataFrame, split: str) -> None:
    """Dumbbell: all test tracks vs only tracks where the stem is actually present."""
    conds = SINGLE_STEMS
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.4), sharey=True)
    # prevalence belongs on the row label, not floating in the plot where it collided
    frac = (ss.set_index("condition")["frac_absent"].groupby(level=0).first()
              .reindex(conds))
    ylabels = [f"{PRETTY[c]}\n{frac[c]*100:.0f}% absent" for c in conds]

    for ax, (target, color) in zip(axes, [("arousal", S_AROUSAL), ("valence", S_VALENCE)]):
        d = ss[ss.target == target].set_index("condition").reindex(conds).reset_index()
        y = np.arange(len(conds))[::-1]
        ax.hlines(y, d["r2_all"], d["r2_present_only"], color=BASELINE, lw=1.6, zorder=2)
        ax.scatter(d["r2_all"], y, s=34, facecolor=SURFACE, edgecolor=color,
                   linewidth=1.4, zorder=3, label="all test tracks")
        ax.scatter(d["r2_present_only"], y, s=34, color=color,
                   edgecolor=SURFACE, linewidth=1.0, zorder=4, label="stem present only")
        ax.set_yticks(y)
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlabel("test $R^2$")
        ax.set_title(target, loc="left", fontsize=10, color=INK)
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.margins(x=0.16, y=0.18)

    handles = [
        Line2D([], [], marker="o", ls="none", markerfacecolor=SURFACE,
               markeredgecolor=INK2, markeredgewidth=1.4, markersize=6,
               label="all test tracks"),
        Line2D([], [], marker="o", ls="none", color=INK2, markersize=6,
               label="stem present only"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, -0.04),
               ncol=2, fontsize=8)

    fig.tight_layout(rect=(0, 0.02, 1, 1.0))
    save(fig, f"fig5_silent_sensitivity_{split}")


# --- figure 6: RQ4/H4 voice vs words -------------------------------------------------
def fig6(res: pd.DataFrame, combo: pd.DataFrame, errcorr: pd.DataFrame, split: str) -> None:
    """Left: R^2 by channel (audio / text / both). Right: the H4 test itself - does
    combining beat the better single channel? Filled marker = CI excludes zero.
    """
    channels = ["audio", "text", "both"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.9))
    ax = axes[0]
    x = np.arange(len(channels))
    w = 0.34
    top = 0.0
    for i, (target, color) in enumerate([("arousal", S_AROUSAL), ("valence", S_VALENCE)]):
        d = res[res.target == target].set_index("channel").reindex(channels).reset_index()
        off = (i - 0.5) * w
        e = err(d)
        ax.bar(x + off, d["value"], w, color=color, label=target,
               edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.errorbar(x + off, d["value"], yerr=e, fmt="none",
                    ecolor=INK2, elinewidth=0.9, capsize=2.5, zorder=4)
        for xi, v, hi in zip(x + off, d["value"], d["value"] + e[1]):
            ax.text(xi, hi + 0.01, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2, zorder=5)
        top = max(top, float((d["value"] + e[1]).max()))
    ax.set_xticks(x)
    ax.set_xticklabels(["vocal audio", "lyric text", "both"])
    ax.set_ylim(0, top * 1.18)
    ax.set_ylabel("test $R^2$  (95% CI)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=8.5)
    # neutral panel identifiers, not titles -- PeerJ wants no titles inside the image
    ax.set_title("(A)", loc="left", fontsize=10, color=INK, pad=26)

    axb = axes[1]
    d = combo.set_index("target").reindex(["arousal", "valence"]).reset_index()
    y = np.arange(len(d))[::-1]
    colors = [S_AROUSAL, S_VALENCE]
    sig = d["significant"].astype(bool).to_numpy()
    lo = np.clip(d["diff_both_minus_best_single"] - d["ci_lo"], 0, None)
    hi = np.clip(d["ci_hi"] - d["diff_both_minus_best_single"], 0, None)
    axb.axvline(0, color=BASELINE, lw=1.0, zorder=2)
    for yi, di, l, h, c, s in zip(y, d["diff_both_minus_best_single"], lo, hi, colors, sig):
        axb.hlines(yi, di - l, di + h, color=c, lw=1.6, alpha=0.85, zorder=3)
        if s:
            axb.scatter(di, yi, s=40, color=c, edgecolor=SURFACE, linewidth=1.0, zorder=4)
        else:
            axb.scatter(di, yi, s=40, facecolor=SURFACE, edgecolor=c, linewidth=1.4, zorder=4)
    axb.set_yticks(y)
    axb.set_yticklabels([f"{t}\n(vs {b})" for t, b in zip(d["target"], d["best_single"])])
    axb.set_xlabel(r"$\Delta R^2$: both $-$ better single channel")
    axb.set_title("(B)", loc="left", fontsize=10, color=INK)
    axb.xaxis.grid(True, zorder=0)
    axb.set_axisbelow(True)
    axb.margins(x=0.25, y=0.35)

    handles = [
        Line2D([], [], marker="o", ls="none", color=INK2, markersize=6,
               label="CI excludes 0 (combining helps)"),
        Line2D([], [], marker="o", ls="none", markerfacecolor=SURFACE,
               markeredgecolor=INK2, markeredgewidth=1.4, markersize=6,
               label="CI includes 0 (no gain)"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, -0.05),
               ncol=2, fontsize=8)

    fig.tight_layout(rect=(0, 0.03, 1, 1.0))
    save(fig, f"fig6_rq4_voice_vs_words_{split}")


# --- the main results table ---------------------------------------------------------
def table1(met_all: pd.DataFrame, pv_all: pd.DataFrame, featset: str, model: str) -> pd.DataFrame:
    rows = []
    for split in sorted(met_all["split"].unique()):
        met = met_all[(met_all.split == split) & (met_all.featset == featset)
                      & (met_all.model == model)]
        pv = pv_all[(pv_all.split == split) & (pv_all.featset == featset)
                    & (pv_all.model == model)]
        for cond in ALL_CONDS:
            row = {"split": split, "condition": cond}
            for target in ["arousal", "valence", "quadrant"]:
                m = met[(met.target == target) & (met.condition == cond)]
                if m.empty:
                    continue
                r = m.iloc[0]
                lbl = "r2" if target != "quadrant" else "macro_f1"
                row[f"{target}_{lbl}"] = round(float(r["value"]), 3)
                row[f"{target}_ci"] = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                p = pv[(pv.target == target) & (pv.condition == cond)]
                if not p.empty:
                    row[f"{target}_vs_mix"] = round(float(p.iloc[0]["diff"]), 3)
                    row[f"{target}_sig"] = bool(p.iloc[0]["significant"])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--featset", default="both")
    ap.add_argument("--model", default="ridge")
    args = ap.parse_args()
    style()

    met_all = pd.read_csv(OUT / "results_metrics.csv")
    pv_all = pd.read_csv(OUT / "analysis_paired_vs_mix.csv")
    ed_all = pd.read_csv(OUT / "analysis_energy_density.csv")
    ss_all = pd.read_csv(OUT / "analysis_silent_sensitivity.csv")
    rq4_res_path = OUT / "rq4_results.csv"
    have_rq4 = rq4_res_path.exists()
    if have_rq4:
        rq4_res_all = pd.read_csv(rq4_res_path)
        rq4_combo_all = pd.read_csv(OUT / "rq4_combo_vs_best.csv")
        rq4_err_all = pd.read_csv(OUT / "rq4_error_correlation.csv")
    else:
        print("rq4_results.csv not found - skipping fig6 (run scripts/phase7_rq4.py first)")

    def pick(df: pd.DataFrame, split: str) -> pd.DataFrame:
        return df[(df.split == split) & (df.featset == args.featset)
                  & (df.model == args.model)].copy()

    def pick_rq4(df: pd.DataFrame, split: str) -> pd.DataFrame:
        return df[(df.split == split) & (df.model == args.model)].copy()

    for split in ["official", "artist_disjoint"]:
        print(f"\n{split} ({args.featset}/{args.model}):")
        met, pv = pick(met_all, split), pick(pv_all, split)
        if met.empty:
            print("  no rows for this spec; skipped")
            continue
        fig1(met, split)
        fig2(pv, split)
        fig3(met, split)
        fig4(pick(ed_all, split), split)
        fig5(pick(ss_all, split), split)
        if have_rq4:
            fig6(pick_rq4(rq4_res_all, split), pick_rq4(rq4_combo_all, split),
                 pick_rq4(rq4_err_all, split), split)

    t1 = table1(met_all, pv_all, args.featset, args.model)
    t1.to_csv(OUT / "table1_main_results.csv", index=False)
    print(f"\nwrote table1_main_results.csv rows={len(t1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
