#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断：B 组（单行训练）在非训练行上的预测是不是"近似常数"。

目的：把「跨行泛化失败」的机制说清楚 —— B 的输出在 Row 0/15/31 上
到底是"学到但错了"，还是"根本没给信号、退化成常数"。
输出每行：n_valid/角、真值 mean/std、预测 mean/std、bias(pred-true)。

用法：python scripts/diag_row_generalization.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
sys.path.insert(0, str(ROOT / "scripts"))

from dose_eval import (build_stack, predict_phase, norm_of, read_rows,  # noqa: E402
                       scan_root)

CSV_NAME = "sparse_targets_3MeV.csv"
ENERGY = 3.0
ROWS = [0, 15, 16, 31]
A_SEED1 = "20261027_0411_cnn_bs8_lr1e-4_base32_3mev"
B_SEED1 = "20261027_0553_cnn_bs8_lr1e-4_row16_3mev"


def stats(pr: np.ndarray, tgt: np.ndarray, msk: np.ndarray, r: int) -> str:
    p = pr[:, r, :][msk[:, r, :]]
    t = tgt[:, r, :][msk[:, r, :]]
    if p.size == 0:
        return f"Row {r:>2}: 无有效像素"
    return (f"Row {r:>2}: n/角={p.size / pr.shape[0]:7.1f}  "
            f"true(mean={t.mean():+.4f}, std={t.std():.4f})  "
            f"pred(mean={p.mean():+.4f}, std={p.std():.4f})  "
            f"bias={p.mean() - t.mean():+.4f}  "
            f"RMSE={np.sqrt(((p - t) ** 2).mean()):.4f}")


def main() -> int:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_p = ROOT / "config.json"
    mean, std, fill = norm_of(json.loads(cfg_p.read_text(encoding="utf-8")))
    rows_all = read_rows(resolve_csv(CSV_NAME))
    root = scan_root(rows_all)
    rows_e = [r for r in rows_all if abs(float(r["energy_MeV"]) - ENERGY) < 1e-9]
    L, ang, M = build_stack(rows_e, root, fill)
    ph_of = {int(float(r["target_angle_deg"])): r["phase"] for r in rows_e}

    # 评估方向固定 A->B（预测 phase B 的 180 视角），便于两模型对比
    mis_idx = np.array([a for a in range(360) if ph_of[a] == "B"])
    tgt, msk = L[mis_idx], M[mis_idx]

    for tag, rid, single in (("A(32行)", A_SEED1, False), ("B(单行)", B_SEED1, True)):
        pr, _ = predict_phase(rid, cfg_p, resolve_csv(CSV_NAME), ENERGY, "B", 8, device,
                              single_row=single)
        print(f"\n=== {tag}  {rid} ===")
        for r in ROWS:
            print("  " + stats(pr, tgt, msk, r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
