#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""临时核对：9 根棒中心圆盘的绝对重建值（不是在差值图上找峰值，而是直接看每根棒）。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import p01_flaw_step3_eval as ev  # noqa: E402

SRC = ROOT / "runs" / f"{ev.RUN_ID}_flaw_step3"
dense = np.load(SRC / "recon" / "dense.npy").astype(np.float64)
cnn = np.load(SRC / "recon" / "cnn_completed_A.npy").astype(np.float64)

yy, xx = np.mgrid[0:1200, 0:1200]


def disc(img, r, az, rad):
    cy = 600.0 - r * math.sin(math.radians(az))
    cx = 600.0 + r * math.cos(math.radians(az))
    m = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad * rad
    return float(img[m].mean())


print(f"{'rod':9s} {'r':>6s} {'az':>6s} {'dense r=15':>11s} {'r=30':>9s} "
      f"{'r=50':>9s} {'cnn r=30':>9s}")
for name, r, az in ev.rod_centers():
    row = [disc(dense, r, az, q) for q in (15, 30, 50)]
    print(f"{name:9s} {r:6.1f} {az:6.1f} {row[0]:11.5f} {row[1]:9.5f} "
          f"{row[2]:9.5f} {disc(cnn, r, az, 30):9.5f}")

# 径向剖面：缺陷棒 vs 一根正常角棒
print("\n径向剖面（dense，每 10 px 一环的环均值）")
for tag, r, az in (("corner1(缺陷)", ev.R_CORNER, 135.0),
                   ("corner0(正常)", ev.R_CORNER, 45.0)):
    cy = 600.0 - r * math.sin(math.radians(az))
    cx = 600.0 + r * math.cos(math.radians(az))
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    prof = []
    for k in range(10):
        m = (d2 >= (10 * k) ** 2) & (d2 < (10 * (k + 1)) ** 2)
        prof.append(float(dense[m].mean()))
    print(f"{tag:16s} " + " ".join(f"{v:.5f}" for v in prof))
