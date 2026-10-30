# Sanitisation applied to this public snapshot

Only the **copies** in this repository were edited; the working repository was not modified. Every edit removes a machine-specific path and nothing else.

- `data/csv_paths.py`: stripped project-root prefix x1
- `runs/20261026_1645_cnn_bs8_lr1e-4_dose0.5/config.json`: stripped project-root prefix x2
- `runs/20261026_1744_cnn_bs8_lr1e-4_dose0.5/config.json`: stripped project-root prefix x2
- `runs/20261026_1835_cnn_bs8_lr1e-4_dose0.5/config.json`: stripped project-root prefix x2
- `runs/20261026_1945_cnn_bs8_lr1e-4_dose0.25/config.json`: stripped project-root prefix x2
- `runs/20261026_2026_cnn_bs8_lr1e-4_dose0.25/config.json`: stripped project-root prefix x2
- `runs/20261026_2106_cnn_bs8_lr1e-4_dose0.25/config.json`: stripped project-root prefix x2
- `runs/20261026_2148_cnn_bs8_lr1e-4_dose0.125/config.json`: stripped project-root prefix x2
- `runs/20261026_2230_cnn_bs8_lr1e-4_dose0.125/config.json`: stripped project-root prefix x2
- `runs/20261026_2311_cnn_bs8_lr1e-4_dose0.125/config.json`: stripped project-root prefix x2
- `runs/_dose_eval_f0.125.json`: stripped project-root prefix x2
- `runs/_dose_eval_f0.25.json`: stripped project-root prefix x2
- `runs/_dose_eval_f0.5.json`: stripped project-root prefix x2
- `runs/_flaw11_multi_summary.json`: stripped project-root prefix x12
- `runs/_flaw12_multi_summary.json`: stripped project-root prefix x12
- `runs/_flaw_multi_summary.json`: stripped project-root prefix x12
- `runs/_verify2_main.json`: stripped project-root prefix x2
- `runs/_verify2_main_zh.json`: stripped project-root prefix x2
- `runs/_verify_main.json`: stripped project-root prefix x2
- `runs/_verify_main_zh.json`: stripped project-root prefix x2
- `scripts/_r54c_p0.py`: stripped project-root prefix x2
- `scripts/dose_finalize.py`: replaced user-home path x1
- `scripts/eval_row16_ablation.py`: replaced user-home path x1
- `scripts/p01_flaw_symmetry_exact.py`: stripped project-root prefix x2
- `scripts/row16_ablation.py`: replaced user-home path x1
- `scripts/verify_pdf_textlayer.py`: stripped project-root prefix x1

Re-run `python scripts/build_submission_repo.py` to regenerate.
