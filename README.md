# When Counting Noise Dominates: An Operating-Point Analysis for CNN Projection Completion in High-Z MeV CT — code release

Release snapshot of the **analysis code** that produced every number and figure in the
manuscript above. Authorship is withheld in line with the double-blind submission; the
`runs/` tree below is the same tree the paper cites.

This is a *reproducibility* release, not a software product: it contains the reference
implementation of the data pipeline, the network, the training loop, the baselines and the
statistics used in the paper, together with the per-run configuration, training log and test
metrics of every run that is quoted. It does **not** contain the Monte-Carlo projections or
any trained weight file.

## 1. What is here

| Path | Contents |
|---|---|
| `train.py` | training loop (masked MSE, per-epoch ceilings, `--single-row` ablation) |
| `data/dataset.py` | `SparseCTDataset`: the 5-channel construction, the `-log` transform, the zero-count fill and the pixel mask — the single definition of the input convention |
| `data/csv_paths.py` | single resolution point for every data table |
| `models/unet.py`, `models/unet1d.py` | network definitions (≈1.95 M parameters for the 5→1 configuration) |
| `scripts/` | every analysis script used for a paper number or figure, including the three statistical studies below |
| `config*.json` | the normalisation triple and all run settings; see `norm.scope_note` for why each dose factor has its own statistics |
| `csv_data/` | the target-triplet tables (angles, file paths, split label) for every experiment; paths are **relative** to the data root |
| `runs/*/config.json`, `train_log.csv`, `test_metrics.json`, `test_per_sample.csv` | the recorded provenance of each run (weights are deliberately absent) |
| `runs/_*.json` | aggregate result files; each paper table/figure traces back to one of these |
| `docs/run_artifacts_spec.md` | the archival convention every `runs/` directory follows |
| `SANITIZATION.md` | machine-specific paths removed from this snapshot (the working copy was not modified) |

### The three statistical studies

| Output | Script | Question it answers |
|---|---|---|
| `runs/_dose_cluster_robust.json` | `scripts/dose_cluster_robust.py` | per-dose-factor cluster-robust confidence intervals on Row 16 |
| `runs/_floor_ablation.json` | `scripts/floor_ablation.py` (uses `scripts/row16_cache.py`) | does the 1.5 MeV result depend on the zero-count fill constant? |
| `runs/_coverage_matched.json` | `scripts/coverage_matched.py` (same cache) | does the 1.5 MeV result survive a common (intersection) pixel mask? |

They share one statistic (`scripts/_paired_boot.py`): an equal-weight paired comparison over
180 `(energy, view)` test observations, with a block bootstrap over 30° sector × energy blocks
(10 000 resamples, seed 20261022).

## 2. Data

The projections are Geant4 simulations of a high-Z fuel-assembly phantom at 1.5, 3.0, 4.5 and
6.0 MeV, 32 detector rows × 1200 columns, 360 views per energy at 1° steps. Each view is one
`.npy` float32 array of shape `(32, 1200)`; every scan directory also carries
`results/empty_proj.npy`, the open-beam reference of that directory, which is what the
`-log((I+ε)/(E+ε))` ratio is taken against.

**The projections themselves are not redistributed here** (several tens of GB, generated on
demand). Regenerating them requires (i) a Geant4 build of the phantom described in the
manuscript Methods, with `energy_scan_<E>MeV_*` directory naming, and (ii) an open-beam run per
directory. The `csv_data/*.csv` tables in this repository give the exact file layout the code
expects — the column names (`target_file`, `ch0_file`, `ch_prev_file`, `ch_next_file`,
`scan_dir`, `energy_MeV`, `target_angle_deg`, `phase`, `split`) are read verbatim by
`data/dataset.py`, so pointing them at your own projections is sufficient.

Lower-dose variants (f = 0.5, 0.25, 0.125) are Poisson-thinned copies of the full-dose arrays
(`scripts/build_dose_variants.py --factors 0.5 0.25 0.125`) and use their **own** normalisation
triple and zero-count fill — see the `norm` block of `config_dose_f*.json`.

## 3. Environment

Measured on the machine that produced the results (2026-10-22): Python 3.12.10, NumPy 2.5.4,
matplotlib 3.11.2, torch 2.11.0+cu128 (CUDA 12.8, cuDNN 91900) on an RTX 4070 Laptop GPU
(8 GB). Training runs comfortably inside 8 GB at batch 8. Every analysis script runs on CPU.

```
pip install -r requirements.txt
```

## 4. Reproducing the headline comparison

```bash
# 1. normalisation statistics + zero-count fill (writes the config's `norm` block)
python scripts/compute_norm_stats.py --csv csv_data/sparse_targets.csv --out config.json

# 2. target triplets (theta-3 deg, theta-1 deg, theta+1 deg inputs for the missing view theta)
python scripts/build_sparse_targets.py           # -> csv_data/sparse_targets.csv
python scripts/build_train_manifest.py

# 3. train (three seeds; the headline run used batch 8, lr 1e-4, Adam, 100 epochs, warmup 50)
python train.py --config config.json --seed 20261022 --epochs 100
python train.py --config config.json --seed 20261023 --epochs 100
python train.py --config config.json --seed 20261024 --epochs 100

# 4. projections -> metrics (Row 16, all-32-row, image domain; linear and cubic baselines included)
python scripts/dose_eval.py --factor 1.0 --csv sparse_targets.csv --config config.json \
    --run-ids <the three runs above>

# 5. statistics quoted in the paper
python scripts/dose_cluster_robust.py
python scripts/row16_cache.py --all          # per-pixel Row-16 cache (GPU, ~5 min total)
python scripts/floor_ablation.py
python scripts/coverage_matched.py
```

Split: seed 20261022 with the per-energy sub-seed `"20261022:<energy>"`; the 360 angles of each
energy are shuffled and split 270/45/45 into train/val/test.

## 5. Conventions worth knowing before reading the numbers

* **Row 16 vs all-32-row.** Row 16 (`RECON_ROW`) is the row that enters the reconstruction and
  is the accuracy endpoint. all-32-row is reported as a *variance diagnostic* only — the edge
  rows carry ~60× less open-beam flux and are dominated by zero-count pixels.
* **Masked vs unmasked.** `rmse_valid` restricts each view to its own valid pixels; the Row-16
  comparisons in the defect and dose sections are stated as unmasked (all 1200 columns). The two
  are not interchangeable and differ by several percent at low flux.
* **Coverage has two definitions.** `coverage = (I_target > 0)` and
  `valid_coverage = (I_target > 0) & (empty_proj > 0)`; `coverage ≥ valid_coverage` always.
  Quoting one as the other is a common error in this codebase's history.
* **Each dose factor has its own fill.** 5.50 / 2.80 / 2.30 / 1.60 for f = 1.0 / 0.5 / 0.25 /
  0.125 (p99.9 of the `-log` distribution of that factor's own training split).

## 6. License

MIT — see `LICENSE`.
