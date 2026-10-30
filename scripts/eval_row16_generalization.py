#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""跨行泛化评估（Row-16-only 消融的补充证据）。

用 B 组（Row-16-only 训练）的 3 种子 best.pt，在 3.0 MeV 上对探测器行
{Row 0, Row 15, Row 16, Row 31} 分别计算 RMSE，与 A 组（32 行训练）的同口径 RMSE 对比，
看 B 组学到的角度域结构能否泛化到非训练行。

协议（刻意与现有 Row16 评测完全一致，保证 Row 16 列能复现 0.0868 / 0.0966）
-------------------------------------------------------------------------
- 复用 dose_eval.py 的 predict_phase / build_stack / rmse_pair；
- 预测 opposite 相位的 180 个视角，两方向（A→B, B→A）各算一次，
  再对全部 360 角度求均值（与 _eval_row16_A/B.json 同一口径）；
- 指标为 -log 物理域 per-pixel mask RMSE（mask = I_target>0 & empty_proj>0）；
- 不重训：直接读 6 份 best.pt。

判读
----
- 若 B 在 Row 0/15/31 上不差于（或优于）A：单行训练学到了角度域结构，跨行泛化好，
  Row 16 的优势不是行特定的。
- 若 B 在 Row 0/15/31 上明显差于 A（相对差 > +5%）：单行训练确实丢失了跨行冗余，
  32 行训练是必要的，Row 16 的优势是「行特定的过拟合」。

用法
----
    python scripts/eval_row16_generalization.py
产出：runs/_row16_generalization.json 、 runs/_row16_generalization.md
"""
from __future__ import annotations

import csv
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

from dose_eval import (build_stack, predict_phase, rmse_pair, norm_of,  # noqa: E402
                       read_rows, scan_root)  # noqa: E402

INDEX = RUNS / "_row16_ablation_index.csv"
CSV_NAME = "sparse_targets_3MeV.csv"
ENERGY = 3.0
ROWS = [0, 15, 16, 31]
BATCH = 8


def group_run_ids(group: str) -> list[str]:
    if not INDEX.is_file():
        raise SystemExit(f"[FATAL] 缺 {INDEX}")
    out = [r["run_id"] for r in csv.DictReader(open(INDEX, newline="", encoding="utf-8"))
           if r["group"] == group]
    if len(out) != 3:
        raise SystemExit(f"[FATAL] {group} 只有 {len(out)} 个 run，期望 3")
    return out


def eval_group(run_ids: list[str], single_row: bool, device) -> dict:
    """返回 {row: {'per_seed':[3], 'mean','std','per_dir':[[2],[2],[2]]}}。"""
    cfg_p = ROOT / "config.json"
    mean, std, fill = norm_of(json.loads(cfg_p.read_text(encoding="utf-8")))
    rows_all = read_rows(resolve_csv(CSV_NAME))
    root = scan_root(rows_all)
    rows_e = [r for r in rows_all if abs(float(r["energy_MeV"]) - ENERGY) < 1e-9]
    if len(rows_e) != 360:
        raise SystemExit(f"[FATAL] {CSV_NAME} 的 {ENERGY} MeV 只有 {len(rows_e)} 行，期望 360")
    L, ang, M = build_stack(rows_e, root, fill)
    ph_of = {int(float(r["target_angle_deg"])): r["phase"] for r in rows_e}

    per_seed_rows = {r: [] for r in ROWS}   # 每种子：两方向均值
    per_seed_dirs = {r: [] for r in ROWS}   # 每种子：保留两方向明细
    for rid in run_ids:
        dir_vals = {r: [] for r in ROWS}
        for _in_phase, missing in (("A", "B"), ("B", "A")):
            mis_idx = np.array([a for a in range(360) if ph_of[a] == missing])
            pr, prws = predict_phase(rid, cfg_p, resolve_csv(CSV_NAME), ENERGY, missing,
                                     BATCH, device, single_row=single_row)
            tgt, msk = L[mis_idx], M[mis_idx]
            a_got = np.array([int(float(r["target_angle_deg"])) for r in prws])
            if not np.array_equal(a_got, mis_idx):
                raise SystemExit(f"[FATAL] {rid} 预测角度与目标角度不对齐")
            for r in ROWS:
                (rmse_valid, _rmse_full), _n = rmse_pair(pr, tgt, msk, r)
                dir_vals[r].append(rmse_valid)
        for r in ROWS:
            vals = [x for x in dir_vals[r] if np.isfinite(x)]
            if len(vals) != 2:
                raise SystemExit(f"[FATAL] {rid} Row {r} 有方向未产生有限 RMSE：{dir_vals[r]}")
            per_seed_rows[r].append(float(np.mean(vals)))
            per_seed_dirs[r].append([float(x) for x in dir_vals[r]])
    out = {}
    for r in ROWS:
        arr = np.array(per_seed_rows[r])
        out[r] = {"per_seed": [round(float(x), 6) for x in per_seed_rows[r]],
                  "mean": float(arr.mean()),
                  "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                  "per_dir": per_seed_dirs[r]}
    return out


def main() -> int:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    A = eval_group(group_run_ids("base32_3mev"), single_row=False, device=device)
    B = eval_group(group_run_ids("row16_3mev"), single_row=True, device=device)

    table = {}
    for r in ROWS:
        a_mean, b_mean = A[r]["mean"], B[r]["mean"]
        rel = (b_mean - a_mean) / a_mean * 100.0 if a_mean else float("nan")
        table[r] = {"A": a_mean, "B": b_mean, "rel_pct": rel,
                    "A_per_seed": A[r]["per_seed"], "B_per_seed": B[r]["per_seed"],
                    "A_std": A[r]["std"], "B_std": B[r]["std"]}

    # ---- 判读（只看非训练行 0/15/31） ----
    others = [r for r in ROWS if r != 16]
    rels = [table[r]["rel_pct"] for r in others]
    worst = max(rels)
    best = min(rels)
    if worst < 5.0:
        verdict = ("B 在 Row 0/15/31 上均不差于 A（最差相对差 {:.1f}%）："
                   "Row-16-only 训练学到的角度域结构可跨行泛化，"
                   "Row 16 的优势不是行特定的过拟合，支持「单行训练亦可」的结论。").format(worst)
        code = "generalizes"
    elif worst < 15.0:
        verdict = ("B 在部分非训练行上轻度退化（最差相对差 {:.1f}%，最好 {:.1f}%）："
                   "跨行泛化存在但非完全行无关，需结合 all32 口径判断是否保留 32 行训练。").format(worst, best)
        code = "partial"
    else:
        verdict = ("B 在 Row 0/15/31 上明显差于 A（最差相对差 {:.1f}%）："
                   "Row-16-only 训练丢失了跨行冗余，32 行训练是必要的，"
                   "Row 16 的优势是「行特定的过拟合」。").format(worst)
        code = "overfit_row"

    out = {
        "energy_MeV": ENERGY,
        "rows": [f"row{r}" for r in ROWS],
        "protocol": ("predict opposite phase (A->B, B->A), mean over 360 angles; "
                     "-log physical-domain per-pixel mask RMSE"),
        "A_base32_3mev": A,
        "B_row16_3mev": B,
        "table": {f"row{r}": table[r] for r in ROWS},
        "verdict": {"code": code, "text": verdict,
                    "rel_others_pct": {f"row{r}": table[r]["rel_pct"] for r in others}},
    }
    jp = RUNS / "_row16_generalization.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] {jp}")
    write_md(out, RUNS / "_row16_generalization.md")
    print(f"[md]   {RUNS / '_row16_generalization.md'}")

    print("\n=== 跨行泛化（3.0 MeV，per-row mask RMSE，-log 物理域）===")
    print(f"{'行':>7}{'A(32行)':>12}{'B(单行)':>12}{'相对差':>12}")
    for r in ROWS:
        t = table[r]
        print(f"{'Row '+str(r):>7}{t['A']:>12.6f}{t['B']:>12.6f}{t['rel_pct']:>+11.2f}%")
    print(f"\n判定({code}): {verdict}")
    return 0


def write_md(out: dict, path: Path) -> None:
    t = out["table"]
    L = ["# 跨行泛化评估（Row-16-only 消融补充证据）\n",
         "> 协议与现有 Row16 评测完全一致：预测 opposite 相位 180 视角，两方向（A→B, B→A）"
         "各算一次并对全部 360 角度求均值；指标为 -log 物理域 per-pixel mask RMSE。"
         "**不重训**，直接读 6 份 best.pt。\n",
         "\n## 对比表（3.0 MeV，per-row RMSE）\n",
         "| 行 | A 组（32 行训练） | B 组（单行训练） | 相对差 |",
         "|---|---|---|---|"]
    for r in (0, 15, 16, 31):
        k = f"row{r}"
        tt = t[k]
        sign = "B 好" if tt["rel_pct"] < 0 else "A 好"
        L.append(f"| Row {r} | {tt['A']:.6f} | {tt['B']:.6f} | "
                 f"{tt['rel_pct']:+.2f}% ({sign}) |")
    L.append("\n## 逐种子明细（3 种子，两方向均值）\n")
    L.append("| 行 | 组 | seed1 | seed2 | seed3 | 均值 |")
    L.append("|---|---|---|---|---|---|")
    for r in (0, 15, 16, 31):
        for grp, gk in (("A", "A_base32_3mev"), ("B", "B_row16_3mev")):
            ps = out[gk][r]["per_seed"]
            mn = out[gk][r]["mean"]
            L.append(f"| Row {r} | {grp} | {ps[0]:.6f} | {ps[1]:.6f} | {ps[2]:.6f} | {mn:.6f} |")
    v = out["verdict"]
    L.append(f"\n## 判读\n\n- 代码：`{v['code']}`\n- {v['text']}\n")
    L.append("- 注：Row 16 是 B 组的训练行（相对差已见于三方对比，此处一并列出以便横比）。\n")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
