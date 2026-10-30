#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/p02_oracle_floor.py —— "完美补全"的误差地板（oracle floor）

要回答的问题
------------
图 (b)(c) 里所有方法的绝对误差都随跨度 s 上升。这到底是**补全质量**的问题，
还是**角度采样本身**的问题？

做法：把待补视角直接填**真值**（而不是预测值），再走完全相同的 FBP 流程。
这样得到的就是"在这个视角网格下，即使补全算法完美，误差也至少有这么大"。
若 CNN 的 rmse 已经很接近这条地板，说明误差上升几乎全部来自采样几何，
模型本身已经接近该网格的信息上限。

口径
----
  · 视角集合与 p02_sweep_eval.py 完全一致：sel = sorted(meas ∪ targ)，共 720/s 个视角
  · 参考仍是 dense = 360 视角全实测 hamming FBP
  · s=2 时 sel 就是全部 360 视角 → 地板恒为 0（构造使然，不是 bug）
  · 图像域指标复用 baseline_recon.image_metrics（同一 FOV 圆、同一 SSIM 裁剪、同一 psnr_common）

输出
----
  runs/_reviewer_p0/oracle_floor.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import baseline_recon as br                        # noqa: E402

OUT = ROOT / "runs" / "_reviewer_p0" / "oracle_floor.json"
FILTER = "hamming"
ENERGIES = ("1.5", "3", "4.5", "6")
STEPS = (2, 4, 6, 8)


def main() -> int:
    cfg = json.loads((br.RUNS_DIR / "20261024_0323_cnn_bs8_lr1e-4" / "config.json")
                     .read_text(encoding="utf-8"))
    fill = float(cfg["norm"]["zero_log_fill"])

    sino = {e: br.build_sinogram(e, ROOT / "3_3", fill) for e in ENERGIES}
    dense = {e: br.restrict_to_fov(br.recon_fbp(sino[e][0], sino[e][1], FILTER))
             for e in ENERGIES}
    dr_common = max(float(dense[e].max() - dense[e].min()) for e in ENERGIES)
    mask = br.fov_mask()

    res: dict = {"filter": FILTER,
                 "dense_reference": "360 视角全实测 hamming FBP",
                 "data_range_common": dr_common,
                 "note": "s=2 时 oracle 视角集合 = 全部 360 视角，地板恒为 0（构造使然）",
                 "per_step": {}}

    for s in STEPS:
        h = s // 2
        per_e: dict = {}
        for e in ENERGIES:
            row16, theta = sino[e]
            rows = []
            for c in range(s):
                meas = [a for a in range(360) if a % s == c]
                targ = [a for a in range(360) if a % s == (c + h) % s]
                sel = sorted(meas + targ)
                img = br.restrict_to_fov(br.recon_fbp(row16[sel],
                                                      np.array(sel, dtype=float), FILTER))
                rows.append(br.image_metrics(dense[e], img, mask))
            mse = float(np.mean([m["rmse_valid"] ** 2 for m in rows]))
            per_e[e] = {
                "n_directions": len(rows),
                "n_views": 720 // s,
                "rmse_valid": math.sqrt(mse),
                "rmse_full": float(np.mean([m["rmse_full"] for m in rows])),
                # s=2 时地板恰为 0（视角集合 = 全部 360 视角），PSNR 记 inf
                "psnr_common": float(np.mean(
                    [float("inf") if m["rmse_valid"] == 0.0
                     else 10.0 * math.log10(dr_common ** 2 / (m["rmse_valid"] ** 2))
                     for m in rows])),
                "ssim": float(np.mean([m["ssim"] for m in rows])),
            }
        mse_all = float(np.mean([v["rmse_valid"] ** 2 for v in per_e.values()]))
        res["per_step"][str(s)] = {
            "n_completed_views": 720 // s,
            "per_energy": per_e,
            "overall": {"rmse_valid": math.sqrt(mse_all),
                        "psnr_common": float(np.mean([v["psnr_common"] for v in per_e.values()])),
                        "ssim": float(np.mean([v["ssim"] for v in per_e.values()]))},
        }
        o = res["per_step"][str(s)]["overall"]
        print(f"[s={s}] 完美补全地板：rmse={o['rmse_valid']:.6f} "
              f"psnr_c={o['psnr_common']:.4f} ssim={o['ssim']:.4f} "
              f"({720 // s} 视角)", flush=True)

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
