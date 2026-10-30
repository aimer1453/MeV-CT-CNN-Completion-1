#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/grp5_paired_stats.py —— grouped split（近角训练 / 远角外推）5 种子的
Row-16 投影域统计：均值、SD、逐视角配对检验

口径（与 `_flaw12_span_table.md` / `r34_split_attribution.md` §1 一致）
------------------------------------------------------------------
- split : `sparse_targets_3MeV_grouped.csv`（0–179° 训练/验证，180–359° 全部作 test）
- 域   : 投影域 Row 16（`br.RECON_ROW`），mask = 该视角该行的 `(I>0)&(E>0)`
- 指标 : 逐视角 RMSE（`sqrt(mean((pred-gt)^2))` 只在 mask 内），seed = 20261022…26
- 对照 : linear = 0.5*(L[θ-h] + L[θ+h])，h = 1（s = 2°）
- 归一化：各 run config 自带（主域 0.7202/0.9249/5.5）

为什么用配对检验
----------------
CNN 与 linear 在**同一批 180 个视角**上评估，误差天然成对。逐视角配对检验
（Wilcoxon signed-rank 为主，paired t 为辅）回答的是"在这些视角上 CNN 是否系统性更差"。
但视角之间并非独立（同一物体、同一几何），所以视角级 p 值**只说明方向一致性**，
不能当作独立重复的统计证据；真正独立的重复单位是种子（n = 5）。
本脚本因此同时报：(a) 5 个种子的 ratio 全部 > 1 的事实，(b) 视角级配对 p 值，
(b) 必须在正文里注明"views are not independent replicates"。

用法
----
    python scripts/grp5_paired_stats.py
    python scripts/grp5_paired_stats.py --runs A B C D E --out-md runs/_x.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
P0 = ROOT / "runs" / "_reviewer_p0"

DEFAULT_RUNS = [
    "20261029_0217_cnn_bs8_lr1e-4_grouped_s22",
    "20261029_0217_cnn_bs8_lr1e-4_grouped_s23",
    "20261029_0225_cnn_bs8_lr1e-4_grouped_s24",
    "20261030_0202_cnn_bs8_lr1e-4_grouped_s25",
    "20261030_0211_cnn_bs8_lr1e-4_grouped_s26",
]


def load_run(rid: str) -> dict:
    p = P0 / f"projonly_flaw_s2_{rid}.json"
    if not p.is_file():
        raise SystemExit(f"缺少 {p}；先跑 p02_fault_span_eval.py --csv sparse_targets_3MeV_grouped.csv")
    j = json.loads(p.read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "runs" / rid / "config.json").read_text(encoding="utf-8"))
    pv = j["projection"]["per_view"]
    return {
        "run_id": rid,
        "seed": int(cfg["seed"]),
        "epoch": j["cnn_epoch"],
        "cnn": np.asarray(pv["cnn"]["row16"], dtype=float),
        "lin": np.asarray(pv["linear"]["row16"], dtype=float),
        "cub": np.asarray(pv["cubic"]["row16"], dtype=float),
        "angles": j["projection"]["test_angles_order"],
        "agg_cnn": j["projection"]["cnn"]["row16"]["rmse_valid"],
        "agg_lin": j["projection"]["linear"]["row16"]["rmse_valid"],
        "agg_cub": j["projection"]["cubic"]["row16"]["rmse_valid"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--out-md", default="runs/_grp5_row16_stats.md")
    ap.add_argument("--out-json", default="runs/_grp5_row16_stats.json")
    a = ap.parse_args(argv)

    runs = [load_run(r) for r in a.runs]
    n_ang = len(runs[0]["angles"])
    for r in runs[1:]:
        if r["angles"] != runs[0]["angles"]:
            raise SystemExit(f"{r['run_id']} 的 test 角度集合与第一份不同，不能配对")

    cnn = np.vstack([r["cnn"] for r in runs])      # (n_seed, n_ang)
    lin = np.vstack([r["lin"] for r in runs])
    cub = np.vstack([r["cub"] for r in runs])

    # ---- 种子级 ----
    per_seed = []
    for r in runs:
        per_seed.append({
            "seed": r["seed"], "epoch": r["epoch"],
            "cnn": float(np.sqrt(np.mean(r["cnn"] ** 2))),
            "lin": float(np.sqrt(np.mean(r["lin"] ** 2))),
            "cub": float(np.sqrt(np.mean(r["cub"] ** 2))),
            "ratio": float(np.sqrt(np.mean(r["cnn"] ** 2)) / np.sqrt(np.mean(r["lin"] ** 2))),
        })
    rat = np.array([p["ratio"] for p in per_seed])
    cnn_s = np.array([p["cnn"] for p in per_seed])

    # ---- 视角级配对（种子平均）----
    cnn_m, lin_m, cub_m = cnn.mean(0), lin.mean(0), cub.mean(0)
    d = cnn_m - lin_m
    w = stats.wilcoxon(cnn_m, lin_m, alternative="greater")
    t = stats.ttest_rel(cnn_m, lin_m)
    # 每个种子各自视角级检验（看方向是否稳定）
    per_seed_p = []
    for i in range(len(runs)):
        wi = stats.wilcoxon(cnn[i], lin[i], alternative="greater")
        per_seed_p.append(float(wi.pvalue))

    worse_frac = [float(np.mean((cnn[i] > lin[i]))) for i in range(len(runs))]

    res = {
        "n_seeds": len(runs),
        "n_test_views": n_ang,
        # 边界从 CSV 实核（2026-10-30 12:38）：train 0-89 / val 90-179 / test 180-359
        "split": ("grouped: train 0-89deg (90), val 90-179deg (90), "
                  "test 180-359deg (180); train+val and test disjoint"),
        "domain": "projection row16, masked (I>0)&(E>0)",
        "per_seed": per_seed,
        "seed_level": {
            "cnn_mean": float(cnn_s.mean()), "cnn_sd": float(cnn_s.std(ddof=1)),
            "ratio_mean": float(rat.mean()), "ratio_sd": float(rat.std(ddof=1)),
            "ratio_min": float(rat.min()), "ratio_max": float(rat.max()),
            "n_seeds_ratio_gt1": int((rat > 1).sum()),
        },
        "view_level_paired": {
            "mean_cnn": float(cnn_m.mean()), "mean_lin": float(lin_m.mean()),
            "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "wilcoxon_W": float(w.statistic), "wilcoxon_p": float(w.pvalue),
            "ttest_t": float(t.statistic), "ttest_p": float(t.pvalue),
            "frac_views_cnn_worse": worse_frac,
            "per_seed_wilcoxon_p": per_seed_p,
            "note": "views are NOT independent replicates; use seed-level direction "
                    "(5/5 ratio > 1) as the robustness statement, p-values as consistency only",
        },
    }
    Path(a.out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    L = []
    L.append("# grouped split（近角训练 / 远角外推）5 种子 Row-16 投影域统计\n")
    L.append(f"- split：`sparse_targets_3MeV_grouped.csv`（0–179° 训练/验证，**180–359° 全部 "
             f"{n_ang} 个视角作 test**）")
    L.append("- 域：投影域 Row 16，mask = 该视角该行的 `(I>0)&(E>0)`")
    L.append("- 对照：linear = 0.5·(L[θ−1°] + L[θ+1°])，s = 2°\n")
    L.append("## 逐种子\n")
    L.append("| seed | best epoch | CNN row16 | linear row16 | cubic row16 | **CNN/linear** |")
    L.append("|---|---|---|---|---|---|")
    for p in per_seed:
        L.append(f"| {p['seed']} | {p['epoch']} | {p['cnn']:.6f} | {p['lin']:.6f} | "
                 f"{p['cub']:.6f} | **{p['ratio']:.4f}** |")
    L.append("")
    L.append(f"**种子级汇总（n = {len(runs)}）**：CNN = {cnn_s.mean():.6f} ± {cnn_s.std(ddof=1):.6f}；"
             f"CNN/linear = **{rat.mean():.4f} ± {rat.std(ddof=1):.4f}**"
             f"（范围 {rat.min():.4f}–{rat.max():.4f}，"
             f"**{int((rat > 1).sum())}/{len(runs)} 个种子 ratio > 1**）\n")
    L.append("## 视角级配对检验（{n} 个视角，种子平均后）\n".format(n=n_ang))
    L.append("| 统计量 | 值 |")
    L.append("|---|---|")
    L.append(f"| 平均逐视角 RMSE（CNN / linear） | {cnn_m.mean():.6f} / {lin_m.mean():.6f} |")
    L.append(f"| 平均差（CNN − linear） | {d.mean():+.6f} |")
    L.append(f"| 中位差 | {np.median(d):+.6f} |")
    L.append(f"| Wilcoxon signed-rank（单侧 greater） | W = {w.statistic:.1f}，**p = {w.pvalue:.3e}** |")
    L.append(f"| 配对 t 检验（双侧） | t = {t.statistic:.2f}，p = {t.pvalue:.3e} |")
    L.append(f"| 逐种子 CNN 更差视角占比 | {', '.join(f'{f*100:.1f}%' for f in worse_frac)} |")
    L.append(f"| 逐种子 Wilcoxon p | {', '.join(f'{p:.2e}' for p in per_seed_p)} |")
    L.append("")
    L.append("> **口径警告 1**：视角之间不独立（同一物体、同一几何、相邻角高度相关），")
    L.append("> 因此视角级 p 值只证明**方向一致性**，不可当作独立重复的统计证据。")
    L.append(f"> 真正独立的重复单位是种子（n = {len(runs)}），正文应写 "
             f"\"all {len(runs)} seeds give a ratio between {rat.min():.2f} and {rat.max():.2f}\"，")
    L.append("> 而不是引用视角级 p。")
    L.append("")
    L.append("> **口径警告 2（p 值在此处是退化值）**：5 个种子各自都有 **180/180 个视角差同号**，")
    L.append(f"> Wilcoxon 统计量因此恒等于 n(n+1)/2 = {n_ang * (n_ang + 1) // 2}"
             f"（所有种子逐位相同，p 也逐位相同，"
             f"{per_seed_p[0]:.3e}）。这意味着该 p **只编码了\"全部同号\"这一件事**，")
    L.append("> 与效应大小无关，不能作为\"高度显著\"的证据引用；效应大小请看种子间比值的区间。")
    Path(a.out_md).write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"[OK] {a.out_md}\n[OK] {a.out_json}")
    print(f"  CNN/linear = {rat.mean():.4f} ± {rat.std(ddof=1):.4f}  "
          f"({int((rat > 1).sum())}/{len(runs)} > 1)")
    print(f"  view-level wilcoxon p = {w.pvalue:.3e}, frac worse = {worse_frac}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
