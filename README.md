# Where Musical Emotion Lives

Analysis code for *"Arousal is envelope, valence is structure: a source-separation study of where musical emotion lives"* (Balajee Kalyanasundaram, submitted to PeerJ Computer Science).

This repository holds the pipeline only. The derived data it produces — features, per-track predictions, analysis tables, and track metadata — is deposited separately on Zenodo at DOI [10.5281/zenodo.21866703](https://doi.org/10.5281/zenodo.21866703), which always resolves to the newest version.

## Description

Music emotion recognition usually treats a recording as one undifferentiated signal. This project separates every clip of a music-emotion corpus into four instrumental sources, rebuilds eleven listening conditions from those sources, and asks where arousal and valence information actually lives — distinguishing **sufficiency** (a source predicts the label on its own) from **necessity** (removing a source costs the model).

The pipeline is eight scripts run in order. Each writes a persistent artifact that the next consumes, so any stage can be re-run without repeating the one before it. Separation is a multi-hour job and is resumable at song granularity; feature extraction is resumable at shard granularity.

## Dataset Information

**Third-party source data (not redistributed here).**

| Item | Value |
|---|---|
| Dataset | MERGE Audio Balanced |
| Source | Zenodo, https://doi.org/10.5281/zenodo.13939205 |
| Citation | Louro PL, Redinho H, Santos R, Malheiro R, Panda R, Paiva RP. 2024. MERGE dataset (version 1.1) [data set]. Zenodo. |
| Companion paper | Louro et al. 2026, *IEEE Transactions on Affective Computing* 17(2):2058–2075, DOI 10.1109/TAFFC.2026.3668037 |
| License | Creative Commons Attribution-NonCommercial 4.0 International |
| Contents used | 3,232 audio clips (~30 s each), 808 per Russell quadrant; continuous arousal and valence values; quadrant labels Q1–Q4; artist metadata; lyrics for a 2,075-track subset |

Obtain the audio from Zenodo under its own license terms. **Neither this repository nor the Zenodo deposit redistributes audio or lyric text** — only derived numerical artifacts, which is what the MERGE license permits.

**Derived data released with the paper**, in the Zenodo deposit above (see its `MANIFEST.csv` for byte counts and SHA-256 checksums):

| File | Contents |
|---|---|
| `stem_features.parquet` | Per-track features for the 11 loudness-normalized conditions (3,232 × 11 = 35,552 rows). Primary analysis input |
| `stem_features_raw.parquet` | The same at natural stem level, no loudness normalization (Control C1) |
| `stem_features_shuffle.parquet` | Features for the 5 spectral-shuffle null conditions (Control C3) |
| `lyrics_features.parquet` | Sentence embeddings for the 2,075-track lyrics subset plus word counts. Embeddings only — no lyric text |
| `predictions.zip` | Per-track test-set predictions for every split × feature set × model × condition × target |
| `analysis_tables.zip` | All analysis outputs, including Tables 1–5 |
| `metadata.zip` | Track manifest (IDs, arousal/valence, quadrant, artist, both split assignments) and the separation log. Lyric text and absolute paths removed |
| `code.zip` | The pipeline scripts described below |

The released features make it possible to reproduce every number in the paper **without repeating separation**, which is the expensive step.

## Code Information

| Phase | Script | Output |
|---|---|---|
| 0 | `scripts/bench_separation.py` | Separator benchmark used to choose the model |
| 1 | `scripts/build_manifest.py` | `outputs/manifest.parquet` |
| 2 | `scripts/separate_stems.py` | `stems/<song_id>/<stem>.flac`, `outputs/separation_log.parquet` |
| 3 | `scripts/extract_features.py` | `outputs/stem_features.parquet` (+ `_raw`, `_shuffle`) |
| 4 | `scripts/phase4_models.py` | `outputs/results_metrics.csv`, `outputs/test_predictions_{reg,clf}.parquet` |
| 5 | `scripts/phase5_analysis.py` | `outputs/analysis_*.csv` |
| 6 | `scripts/phase6_figures.py` | `outputs/figures/fig1-6_*.{pdf,png}`, `outputs/table1_main_results.csv` |
| 7 | `scripts/phase7_rq4.py` | `outputs/rq4_*.csv`, `outputs/rq4_test_predictions.parquet` |
| 8 | `scripts/phase8_tables.py` | `outputs/table{2,3,4,5}_*.csv` |

Supporting modules: `scripts/conditions.py` (the eleven condition definitions), `scripts/features.py` (the feature extractors), `scripts/extract_lyrics.py` (lyrics subset for the RQ4 sub-study), `scripts/test_phase4_synthetic.py` (a synthetic-data check on the modeling code).

## Usage Instructions

Run the phases in order from the repository root. Phases 4–8 read only on-disk artifacts, so starting from the released `stem_features.parquet` reproduces every published number without running phases 0–3.

```
python scripts/build_manifest.py                      # phase 1
python scripts/separate_stems.py --threads 12         # phase 2  (~5 h, CPU, resumable)
bash scripts/run_phase3.sh                            # phase 3  (features; W/T/B tune workers)
python scripts/phase4_models.py --models ridge gbm    # phase 4
python scripts/phase5_analysis.py                     # phase 5
python scripts/phase6_figures.py                      # phase 6
python scripts/phase7_rq4.py                          # phase 7
python scripts/phase8_tables.py                       # phase 8
```

Reproducing the analysis from the released features only:

```
python scripts/phase4_models.py --features stem_features.parquet --models ridge gbm
python scripts/phase5_analysis.py
python scripts/phase7_rq4.py
python scripts/phase8_tables.py
```

**Phase 1 has an extra dependency and is not reproducible from this deposit alone.** `build_manifest.py` resolves MERGE's three ID-linking schemes by reading `outputs/merged_dataset.parquet` from a prior study by the same author, together with MERGE's `tvt_dataframes` split files and the audio. All three are located under `PRIOR_STUDY_ROOT`, which defaults to a sibling `music_research/` checkout. The prior study's parquet has not been publicly released, so **external reproducers should skip phase 1**: the sanitised `manifest.parquet` in `metadata.zip` is exactly phase 1's output, and phases 2–8 read only `outputs/`. Copy it to `outputs/manifest.parquet` and continue from phase 2 (or, to skip separation entirely, from phase 4 using the released `stem_features.parquet`).

The Control C3 null has its own driver, `scripts/run_shuffle_pipeline.bat`, which extracts features for the shuffled conditions and re-runs phases 4–5 under the `shuffle` tag.

Useful flags: `phase4_models.py --featsets/--splits/--boot/--tag/--features`; `phase5_analysis.py --tag/--primary-featset/--primary-model`; `phase6_figures.py --featset/--model`; `separate_stems.py --limit/--threads`; `extract_features.py --stage {handcrafted,clap,assemble} --no-loudnorm --shuffle`.

The audio is expected at the path recorded in the manifest, and separated stems under `stems/`.

## Requirements

Python 3.12 (developed on 3.12.10), CPU-only — no GPU is required, and the separator was chosen for CPU throughput.

| Package | Version used |
|---|---|
| numpy | 1.26.4 |
| pandas | 2.2.3 |
| scipy | 1.14.1 |
| scikit-learn | 1.8.0 |
| librosa | 0.10.2.post1 |
| soundfile | 0.13.1 |
| pyloudnorm | loudness normalization |
| torch | 2.10.0+cpu |
| torchaudio | 2.11.0+cpu |
| demucs | 4.1.0 |
| transformers, with the LAION-CLAP checkpoint `laion/larger_clap_music` | feature extraction |
| sentence-transformers, `all-mpnet-base-v2` | RQ4 lyrics embeddings only |
| matplotlib | 3.9.2 |

Reference hardware: Intel i7-1360P, 12 cores, CPU-only PyTorch. Separation runs about 5 h for the full corpus at these settings.

## Methodology

1. **Separation.** Every clip is split into four stems (drums, bass, other, vocals) with Hybrid Demucs v3 trained with extra data (`hdemucs_mmi`), `shifts=0`, `overlap=0.25`, no test-time augmentation, so the run is deterministic. Stems are written as 16-bit stereo FLAC at 44.1 kHz.
2. **Conditions.** Eleven per track: the mix (the reference condition), four isolated stems (sufficiency), four leave-one-stem-out remixes (necessity), the stem sum, and the residual. Leave-one-out conditions are the **sum of the kept stems**, never mix-minus-stem, so that no condition inherits the separator's residual.
3. **Loudness.** Every condition is normalized to −23 LUFS before feature extraction, so no condition can win on level alone (Control C1). The un-normalized pass is retained as that control.
4. **Features.** An identical 646-dimensional budget per condition: 134 hand-crafted librosa descriptors plus a 512-dimensional CLAP embedding, mean-pooled over fixed non-overlapping 10 s windows. Feature parity is Control C5.
5. **Models.** Ridge regression (arousal, valence) and multinomial logistic regression (quadrant) as the primary specification; histogram gradient boosting as a secondary robustness check. Regularisation is tuned on validation only; the test split is scored once.
6. **Splits.** The official split, plus an **artist-disjoint** split, because 70.5% of official-split test tracks share an artist with training data.
7. **Inference.** Percentile bootstrap (1,000 resamples) for a single condition; paired bootstrap (2,000 resamples) for every comparison between conditions, since the same tracks appear in all of them.
8. **Controls.** C1 loudness, C2 reconstruction, C3 spectral-shuffle null, C4 near-absent stems, C5 feature parity.
9. **RQ4 sub-study.** Vocal acoustics versus lyric semantics on the 2,075-track lyrics subset.

All random seeds are fixed: seed 42 for model fitting, split construction, and bootstrap resampling; per-song deterministic seeds for the C3 shuffle.

## Citations

If you use this code or the derived data, please cite the paper and the underlying dataset:

- Kalyanasundaram B. *Arousal is envelope, valence is structure: a source-separation study of where musical emotion lives.* Submitted to PeerJ Computer Science.
- Louro PL, Redinho H, Santos R, Malheiro R, Panda R, Paiva RP. 2024. MERGE dataset (version 1.1) [data set]. Zenodo. DOI 10.5281/zenodo.13939205.
- Louro PL, Redinho H, Santos R, Malheiro R, Panda R, Paiva RP. 2026. MERGE: a bimodal audio-lyrics dataset for static music emotion recognition. *IEEE Transactions on Affective Computing* 17(2):2058–2075. DOI 10.1109/TAFFC.2026.3668037.

Principal tools: Défossez (2021) for Hybrid Demucs, McFee et al. (2015) for librosa, Wu et al. (2023) for LAION-CLAP, Pedregosa et al. (2011) for scikit-learn, Efron & Tibshirani (1993) for the bootstrap. Full references are in the paper.

## License & Contribution Guidelines

- **Source dataset:** MERGE Audio Balanced is third-party data under Creative Commons Attribution-NonCommercial 4.0 International. It is used here under those terms, and no audio or lyric text is redistributed.
- **Derived data** in the Zenodo deposit inherits the non-commercial terms of the source dataset.
- **Code** in this repository is released under the MIT License.

This is the analysis code for a single study, released for verification and reuse rather than as a maintained package. Corrections are welcome; if you extend the work, please cite the paper above.