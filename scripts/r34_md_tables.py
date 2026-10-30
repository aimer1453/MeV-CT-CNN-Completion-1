#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/r34_md_tables.py —— 把 R34 任务 1/2/3/5 的数字渲染成可直接粘贴的 Markdown

输入
----
  runs/_r34_summary.json          各 split 的 1D 主指标与 linear/cubic 基线（r34_report.py）
  runs/_r34_span_sweep.json       跨度扫描 3 种子（r34_span_table.py）
  runs/_r34_angle_difficulty.json 逐角度内在难度（r34_angle_difficulty.py）
  runs/_reviewer_p0/sweep100_s2*.json / runs/_r34_{rand2d,grp2d}_s*.json
                                  2D 架构的投影 row16 + 图像域（r34_grouped_eval.py --cnn-run）

输出
----
  runs/_r34_tables.md  + 控制台
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
P0 = RUNS / "_reviewer_p0"

SPLIT_ORDER = ["random", "contig270", "near45", "rand90", "even90", "grouped"]
SPLIT_LABEL = {
    "random": "random（原主协议）",
    "contig270": "contig270（连续 270）",
    "near45": "near45（扇区 90，test 紧邻）",
    "rand90": "rand90（随机 90，test 同 near45）",
    "even90": "even90（等间隔 90，test 同 near45）",
    "grouped": "grouped（扇区 90，test 远距）",
}
N_TRAIN = {"random": 270, "contig270": 270, "near45": 90, "rand90": 90,
           "even90": 90, "grouped": 90}
TEST_RANGE = {
    "random": "散落全圆（内插 1°）",
    "contig270": "315–359（外推 0–45°）",
    "near45": "90–134（外推 0–44°）",
    "rand90": "90–134（外推 0–44°）",
    "even90": "90–134（外推 0–44°）",
    "grouped": "180–359（外推 90–135°）",
}
IMAGE_GLOB = {
    # 只用 3.0 MeV 单档评估产物（r34_grouped_eval.py --cnn-run）；**不要**混入
    # _reviewer_p0/sweep100_s2.json —— 那份是四档能量池化值，与 1D 支的 3.0 MeV
    # 单档不同口径，混进来会把均值抬到 0.1047 / 图像域 0.001048。
    "random": [str(RUNS / "_r34_rand2d_s*.json")],
    "grouped": [str(RUNS / "_r34_grp2d_s*.json")],
}


def collect_2d(pattern_list: list[str]) -> dict:
    """读 2D 架构的评估产物，返回 {run_id: {"row16":..,"all32":..,"img":..}}。"""
    out: dict = {}
    for pat in pattern_list:
        for f in sorted(glob.glob(pat)):
            try:
                d = json.loads(Path(f).read_text(encoding="utf-8"))
                rid = d["cnn_run_id"]
                out[rid] = {
                    "row16": d["projection"]["cnn"]["row16"]["overall"]["rmse_valid"],
                    "all32": d["projection"]["cnn"]["all32"]["overall"]["rmse_valid"],
                    "img": d["image"]["cnn"]["overall"]["rmse_valid"],
                    "img_lin": d["image"]["linear"]["overall"]["rmse_valid"],
                    "src": Path(f).name,
                }
            except Exception:
                continue
    return out


def ms(v: list[float]) -> tuple[float, float]:
    a = np.array(v, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def pct(base: float, val: float) -> str:
    return f"{(base - val) / base * 100:+.1f}%"


def main() -> int:
    sm = json.loads((RUNS / "_r34_summary.json").read_text(encoding="utf-8"))
    sp_p = RUNS / "_r34_span_sweep.json"
    sp = json.loads(sp_p.read_text(encoding="utf-8")) if sp_p.is_file() else None
    ad_p = RUNS / "_r34_angle_difficulty.json"
    ad = json.loads(ad_p.read_text(encoding="utf-8")) if ad_p.is_file() else None
    img2d = {k: collect_2d(v) for k, v in IMAGE_GLOB.items()}

    L: list[str] = []

    # ---------------- 任务 3 ----------------
    L.append("### 任务 3：全部 split 的完整对比（3.0 MeV，100 ep，Row16 = 成像行）\n")
    L.append("| split | 训练行 | test 角 | test 行 | linear Row16 | CNN Row16 (1D) | vs linear "
             "| CNN Row16 (2D) | vs linear | CNN all32 (2D) | vs linear | 图像域 linear "
             "| 图像域 CNN (2D) | vs linear | 判定 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in SPLIT_ORDER:
        e = sm["splits"].get(name)
        if not e:
            continue
        bl = e.get("baseline", {})
        a1, a2 = e["arms"].get("1d"), e["arms"].get("2d")
        tc = e["test_views_by_energy"]
        n_test = list(tc.values())[0] if tc else 0
        lin16 = bl.get("linear_row16")
        lin32 = bl.get("linear_all32")
        cells = [SPLIT_LABEL[name], str(N_TRAIN[name]), TEST_RANGE[name], str(n_test), f"{lin16:.6f}"]
        cells += ([f"{a1['mean']:.6f} ± {a1['sd']:.6f}（n={a1['n_seeds']}）",
                   pct(lin16, a1["mean"])] if a1 else ["—", "—"])
        d2 = img2d.get(name, {})
        if d2:
            r16, _ = ms([v["row16"] for v in d2.values()])
            r32, _ = ms([v["all32"] for v in d2.values()])
            img, _ = ms([v["img"] for v in d2.values()])
            img_lin = list(d2.values())[0]["img_lin"]
            cells += [f"{r16:.6f}", pct(lin16, r16),
                      f"{r32:.6f}", pct(lin32, r32),
                      f"{img_lin:.6f}", f"{img:.6f}", pct(img_lin, img)]
        else:
            cells += ["—", "—", "—", "—", f"{bl.get('image_linear', float('nan')):.6f}", "—", "—"]
        if a1 and lin16:
            r = (lin16 - a1["mean"]) / lin16 * 100
            cells.append("CNN 赢" if r > 2 else ("打平" if r > -2 else "**CNN 输**"))
        else:
            cells.append("—")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("注 1：`linear Row16` 在五套划分里最大差 0.8%，是 test 集本身难度一致的直接证据。")
    L.append("注 2：`图像域 linear` 与 split 无关（线性补全规则不含训练），故全表同值 0.000943；"
             "图像域 CNN 只在有 2D run 的两个 split 上给出。")
    L.append("注 3：Row16 (2D) 是 2D 架构在成像行上的误差，与 1D 架构同角同 mask，可横比。\n")

    # ---------------- 任务 1 ----------------
    if sp:
        L.append("### 任务 1：跨度扫描（3 种子，4 档能量池化，Row16 = 成像行）\n")
        L.append("| span | n | CNN Row16 | linear Row16 | CNN/linear | CNN/cubic | CNN all32 | CNN/linear (all32) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for s in (2, 4, 6, 8):
            a = sp["spans"][f"s{s}"]["agg"]
            r, r32 = a["row16"], a["all32"]
            L.append(f"| {s}° | {r['n_seeds']} | {r['cnn_mean_sd'][0]:.6f} ± {r['cnn_mean_sd'][1]:.6f} "
                     f"| {r['linear'][0]:.6f} | **{r['ratio_linear_mean_sd'][0]:.4f} ± "
                     f"{r['ratio_linear_mean_sd'][1]:.4f}** | {r['ratio_cubic_mean_sd'][0]:.4f} ± "
                     f"{r['ratio_cubic_mean_sd'][1]:.4f} | {r32['cnn_mean_sd'][0]:.6f} ± "
                     f"{r32['cnn_mean_sd'][1]:.6f} | {r32['ratio_linear_mean_sd'][0]:.4f} ± "
                     f"{r32['ratio_linear_mean_sd'][1]:.4f} |")
        L.append("")
        L.append("逐能量 Row16 的 **CNN/linear**（mean ± SD，3 种子）：\n")
        L.append("| span | 1.5 MeV | 3.0 MeV | 4.5 MeV | 6.0 MeV |")
        L.append("|---|---|---|---|---|")
        for s in (2, 4, 6, 8):
            pe = sp["spans"][f"s{s}"]["row16_per_energy_agg"]
            L.append(f"| {s}° | " + " | ".join(
                f"{pe[e]['ratio_linear_mean_sd'][0]:.4f} ± {pe[e]['ratio_linear_mean_sd'][1]:.4f}"
                for e in ("1.5", "3", "4.5", "6")) + " |")
        L.append("")
        L.append("逐能量 Row16 的 **CNN/cubic**（§5.6 引用的 0.984/0.977/0.986 是这个量）：\n")
        L.append("| span | 1.5 MeV | 3.0 MeV | 4.5 MeV | 6.0 MeV |")
        L.append("|---|---|---|---|---|")
        for s in (2, 4, 6, 8):
            pe = sp["spans"][f"s{s}"]["row16_per_energy_agg"]
            L.append(f"| {s}° | " + " | ".join(
                f"{pe[e]['ratio_cubic_mean_sd'][0]:.4f} ± {pe[e]['ratio_cubic_mean_sd'][1]:.4f}"
                for e in ("1.5", "3", "4.5", "6")) + " |")
        L.append("")

    # ---------------- 任务 5 ----------------
    if ad:
        L.append("### 任务 5：test 集内在难度交叉验证（与模型、与 split 无关的线性插值逐角度 RMSE）\n")
        L.append("| split | test 角范围 | n | linear Row16 @test | 难度百分位 | 角度梯度 "
                 "| 梯度百分位 | test→最近 train 距离（中位/最大） | 被 train 夹住 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for name in SPLIT_ORDER:
            r = ad["splits"].get(name)
            if not r:
                continue
            L.append(f"| {name} | {r['test_angle_range'][0]}–{r['test_angle_range'][1]} | {r['n_test']} "
                     f"| {r['lin_rmse_row16_test_mean']:.6f} | P{r['lin_rmse_difficulty_percentile']:.1f} "
                     f"| {r['ang_grad_test_mean']:.5f} | P{r['ang_grad_percentile']:.1f} "
                     f"| {r['nn_train_dist_median']:.1f}° / {r['nn_train_dist_max']:.1f}° "
                     f"| {r['bracket_rate']*100:.1f}% |")
        L.append(f"\n全圆 360 角的 linear Row16 分布：mean {ad['global']['lin_rmse_row16_mean']:.6f}，"
                 f"SD {ad['global']['lin_rmse_row16_sd']:.6f}，"
                 f"P5–P95 = {ad['global']['lin_rmse_row16_p05']:.6f}–{ad['global']['lin_rmse_row16_p95']:.6f}"
                 f"（{ad['n_draw']} 次随机同规模抽样定标）。\n")

    out = RUNS / "_r34_tables.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[md] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
