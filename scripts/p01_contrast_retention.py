#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/p01_contrast_retention.py —— R12-2：30 epoch 的对比度保留率

定义（与 5 epoch 那次**逐字一致**，否则两批数字不可比）
--------------------------------------------------------
  棒阵对比度  C = p99(I) − p50(I)，统计域为物模区 r ≤ 420 px（圆心 = 图像中心）
  保留率      R = C_method / C_dense      （同一能量、同一相位）

为什么用 p99−p50 而不是 max−min：max 被 1.5 MeV 的孤立热点（像素 (600,600)，
0.2303，是 p99.9 的 9.3 倍）支配，min 在圆外零区取 0，两端都不可用。

CNN 与线性插值各有两个相位产物（A / B）。口径：
  逐相位先算 R(A)、R(B)，再取算术平均 → 与 test_metrics 的 "逐样本再平均" 同构。
  同时输出 "ratio of means" 备查，两者差在 1e-3 量级，正文用前者。

输入
----
  runs/<run_id>_cnn_recon/recon/cnn_<E>_{A,B}.npy
  runs/<run_id>_linear_interp_image/recon/linear_<E>_{A,B}.npy
  runs/<run_id>_fbp_sparse/recon/fbp_<E>_{A,B}.npy          （可选，作参考）
  runs/<run_id>_cnn_recon/recon/dense_<E>_fbp_hamming.npy

输出
----
  runs/_reviewer_p0/contrast_retention_<tag>.json
  docs/figs/contrast_retention/（本脚本只出 JSON 与终端表，不出图）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "_reviewer_p0"
ENERGIES = ("1.5", "3", "4.5", "6")
PHASES = ("A", "B")
R_ROD = 420          # 物模区半径（px），与 5 epoch 那次一致


def rod_mask(shape: tuple[int, int] = (1200, 1200), r: int = R_ROD) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def contrast(img: np.ndarray, m: np.ndarray) -> float:
    v = img[m]
    return float(np.percentile(v, 99) - np.percentile(v, 50))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="20261023_2332_cnn_bs8_lr1e-4")
    ap.add_argument("--tag", default="30ep")
    a = ap.parse_args()

    dcnn = ROOT / "runs" / f"{a.run_id}_cnn_recon" / "recon"
    dlin = ROOT / "runs" / f"{a.run_id}_linear_interp_image" / "recon"
    dfbp = ROOT / "runs" / f"{a.run_id}_fbp_sparse" / "recon"
    for d in (dcnn, dlin):
        if not d.is_dir():
            raise SystemExit(f"缺目录：{d}")

    m = rod_mask()
    res: dict[str, dict] = {}
    print(f"物模区 r ≤ {R_ROD} px，{int(m.sum())} 像素；run = {a.run_id}\n")
    hdr = f"{'E(MeV)':>7} {'dense':>9} {'linear':>9} {'R_lin':>7} {'CNN':>9} {'R_cnn':>7} {'fbp':>9} {'R_fbp':>7}"
    print(hdr); print("-" * len(hdr))

    for e in ENERGIES:
        dense = np.load(dcnn / f"dense_{e}_fbp_hamming.npy")
        cd = contrast(dense, m)
        row: dict = {"dense": cd}
        for name, d, pre in (("linear", dlin, "linear"), ("cnn", dcnn, "cnn"),
                             ("fbp", dfbp, "fbp")):
            if not d.is_dir():
                continue
            per = {}
            for p in PHASES:
                f = d / f"{pre}_{e}_{p}.npy"
                if not f.is_file():
                    continue
                per[p] = contrast(np.load(f), m)
            if len(per) != 2:
                continue
            row[name] = {"per_phase": per,
                         "mean_reler": float(np.mean([v / cd for v in per.values()])),
                         "ratio_of_means": float(np.mean(list(per.values())) / cd),
                         "mean_abs": float(np.mean(list(per.values())))}
        # ---- 相对 dense 的"对比度损失%"
        for name in ("linear", "cnn", "fbp"):
            if name in row:
                row[name]["loss_pct"] = (1.0 - row[name]["mean_reler"]) * 100.0
        res[e] = row
        print(f"{e:>7} {cd:9.5f} "
              f"{row['linear']['mean_abs']:9.5f} {row['linear']['mean_reler']*100:6.1f}% "
              f"{row['cnn']['mean_abs']:9.5f} {row['cnn']['mean_reler']*100:6.1f}% "
              + (f"{row['fbp']['mean_abs']:9.5f} {row['fbp']['mean_reler']*100:6.1f}%"
                 if "fbp" in row else ""))

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"contrast_retention_{a.tag}.json"
    p.write_text(json.dumps({"run_id": a.run_id, "radius_px": R_ROD,
                             "definition": "C = p99 - p50 inside r<=420; R = C/C_dense; "
                                           "per-phase then arithmetic mean",
                             "per_energy": res},
                            ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")
    print(f"\n[OK] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
