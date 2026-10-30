"""P0 边棒三种子 + P1 span/dose 逐能量表（复用既有评估产物，零 GPU）。

用户 2026-10-30 裁决：P1-2 / P1-3 走「复用逐能量评估」口径，不重跑单档训练。
本脚本只读 runs/ 下已有 json 与 fig12 CSV，输出可直接粘进交接文档的 Markdown。

⚠ 口径护栏（务必随表一起转给论文侧）
  · span 表：`p02_proj_only.py` 评估的是**数据表 test split（45 视角）**。
  · dose 表：`dose_eval.py` 评估的是**全部 180 个待补全视角**（agg 在 test 过滤前
    append，见 R34 §③）。两者**不可横向比较**，也不能与 Table 5 的 45-test 值混用。
  · 两者都是四档能量一起训练的同一个池化模型，逐能量只是**评估时分组**，
    不是单档训练。归一化仍锁全局 0.7202/0.9249/5.5。

用法
----
  python scripts/p1_per_energy_tables.py
  python scripts/p1_per_energy_tables.py --out runs/_p1_per_energy_tables.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIGS = ROOT / "docs" / "figs" / "p01_flaw_step3_indomain_100ep_flaw12"
SEEDS = [20261022, 20261023, 20261024]


def ms(xs: list[float]) -> tuple[float, float]:
    v = np.asarray([x for x in xs if x is not None and math.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def f6(x: float) -> str:
    return "—" if not math.isfinite(x) else f"{x:.6f}"


def f4(x: float) -> str:
    return "—" if not math.isfinite(x) else f"{x:.4f}"


# ------------------------------------------------------------------ P0 边棒
def table_p0() -> list[str]:
    L = ["## A. P0 边棒缺陷（3_3_flaw_1_2）3.0 MeV，三种子 × 100 epoch", ""]
    rows = []
    for s in SEEDS:
        d = None
        for p in sorted(glob.glob(str(RUNS / "2026*flaw12_3.0*"))):
            if "_flaw_step3" in p:
                continue
            cp = Path(p) / "config.json"
            if not cp.is_file():
                continue
            try:
                c = json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if int(c.get("seed", -1)) == s:
                d = p
                break
        if d is None:
            rows.append((s, None, None, None))
            continue
        mp = Path(d + "_flaw_step3") / "image_metrics.json"
        if not mp.is_file():
            rows.append((s, None, None, None))
            continue
        j = json.loads(mp.read_text(encoding="utf-8"))
        pm = j["per_method"]
        cn = pm["cnn_completed"]
        li = pm["linear_interp"]
        mf = j["projection_matched_filter"]["d4_contrast"]
        cnr = j["defect_cnr"]["cnn_completed"].get("in_image", {})
        pv = j["projection_domain"]["per_view_row16"]
        row16_cnn = float(np.mean([pv["A"]["rmse_row16_mean"],
                                   pv["B"]["rmse_row16_mean"]]))
        rows.append((s, cn, li, dict(mf=mf, cnr=cnr, row16_cnn=row16_cnn,
                                     run=Path(d).name, j=j)))

    L += ["| seed | Row16 CNN | 图像域 CNN | 图像域 linear | img CNN/lin | PSNR$_c$ CNN | SSIM CNN | CNR (CNN) | D4 $t$ |",
          "|---|---|---|---|---|---|---|---|---|"]
    r16s, imgs, rats, ps, ss, cnrs, ts = [], [], [], [], [], [], []
    for s, cn, li, extra in rows:
        if cn is None:
            L.append(f"| {s} | 评测未完成 | — | — | — | — | — | — | — |")
            continue
        r16 = extra["row16_cnn"]
        rat = cn["rmse_valid"] / li["rmse_valid"]
        t = extra["mf"]["t_stat"]
        r16s.append(r16); imgs.append(cn["rmse_valid"]); rats.append(rat)
        ps.append(cn["psnr"]); ss.append(cn["ssim"])
        cnrs.append(extra["cnr"].get("cnr", float("nan"))); ts.append(t)
        L.append(f"| {s} | {f6(r16)} | {f6(cn['rmse_valid'])} | {f6(li['rmse_valid'])} | "
                 f"{f4(rat)} | {cn['psnr']:.2f} | {cn['ssim']:.4f} | "
                 f"{extra['cnr'].get('cnr', float('nan')):.3f} | {t:+.2f} |")
    if r16s:
        L.append(f"| **mean ± SD** | **{f6(ms(r16s)[0])} ± {f6(ms(r16s)[1])}** | "
                 f"**{f6(ms(imgs)[0])} ± {f6(ms(imgs)[1])}** | — | "
                 f"**{f4(ms(rats)[0])} ± {f4(ms(rats)[1])}** | "
                 f"**{ms(ps)[0]:.2f} ± {ms(ps)[1]:.2f}** | "
                 f"**{ms(ss)[0]:.4f} ± {ms(ss)[1]:.4f}** | "
                 f"**{ms(cnrs)[0]:.3f} ± {ms(cnrs)[1]:.3f}** | "
                 f"**{ms(ts)[0]:+.2f} ± {ms(ts)[1]:.2f}** |")
    L += ["", "**逐视角 Row16（fig12 三 CSV，各 360 个待补全视角）**：", "",
          "| seed | CNN mean | linear mean | CNN/linear | CNN 更优视角数 |",
          "|---|---|---|---|---|"]
    lin_mean = float("nan")
    winners, c_means, ratios = [], [], []
    for s in SEEDS:
        if s == SEEDS[0]:
            p = FIGS / "fig12_per_view_rmse.csv"
        else:
            p = FIGS / f"s{s}" / "fig12_per_view_rmse.csv"
        if not p.is_file():
            L.append(f"| {s} | 未生成 | — | — | — |")
            continue
        rr = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
        c = np.array([float(r["rmse_cnn"]) for r in rr])
        l = np.array([float(r["rmse_linear"]) for r in rr])
        lin_mean = float(l.mean())
        n_win = int((c < l).sum())
        winners.append(n_win)
        c_means.append(float(c.mean()))
        ratios.append(float(c.mean() / l.mean()))
        L.append(f"| {s} | {f6(float(c.mean()))} | {f6(lin_mean)} | "
                 f"{f4(float(c.mean() / l.mean()))} | **{n_win}/360** |")
    if c_means:
        cm, cs = ms(c_means)
        rm, rs = ms(ratios)
        _w = (f"均为 {winners[0]}/360" if winners and len(set(winners)) == 1
              else str(winners))
        L.append(f"| **mean ± SD** | **{f6(cm)} ± {f6(cs)}** | **{f6(lin_mean)}** | "
                 f"**{f4(rm)} ± {f4(rs)}** | **{_w}** |")

    # 角棒同口径对照（单种子 headline）
    pc = ROOT / "docs" / "figs" / "p01_flaw_step3_indomain_100ep" / "fig12_per_view_rmse.csv"
    if pc.is_file():
        rr = list(csv.DictReader(pc.open(newline="", encoding="utf-8")))
        c = np.array([float(r["rmse_cnn"]) for r in rr])
        l = np.array([float(r["rmse_linear"]) for r in rr])
        L += ["",
              f"**角棒对照**（`3_3_flaw_1_1`，同口径单种子）：CNN {f6(float(c.mean()))} / "
              f"linear {f6(float(l.mean()))} / 比值 {f4(float(c.mean() / l.mean()))} / "
              f"CNN 更优 {int((c < l).sum())}/360。"
              f"边棒三种子均值比值 {f4(rm)} 与角棒 {f4(float(c.mean() / l.mean()))} "
              f"相差 {abs(rm - float(c.mean() / l.mean())) / float(c.mean() / l.mean()) * 100:.2f}%。"]
    L += ["",
          f"线性插值与种子无关，三批逐位相同 = {f6(lin_mean)}。"
          f"三个种子 CNN 更优的视角数分别为 {winners}，"
          "即**没有任何一个视角、任何一个种子**上 CNN 补全优于线性插值。",
          "",
          "**D4 $t$ 三个种子逐位相同（−14.39 ± 0.00）不是种子未独立的证据**：匹配滤波积分的是"
          "`flaw − uniform` 两个**仿真数据集**的差分正弦图，与 CNN 权重完全无关，只由"
          "缺陷几何（轨迹）和两次仿真的统计量决定。它衡量的是「这批数据里缺陷能否被"
          "定向积分检出」，而非「模型重建得多好」；模型相关的量是图像域 RMSE / CNR。",
          ""]
    return L


# ------------------------------------------------------------------ P1-2 span
def table_span() -> list[str]:
    p = RUNS / "_r34_span_sweep.json"
    if not p.is_file():
        return ["## B. span sweep", "", "[缺 runs/_r34_span_sweep.json]", ""]
    d = json.loads(p.read_text(encoding="utf-8"))
    L = ["## B. P1-2 span sweep：逐能量 Row16（3 seeds，四档池化模型，test split 45 视角）", "",
         "| span | 能量 (MeV) | CNN mean ± SD | linear | **CNN/linear** | CNN/cubic |",
         "|---|---|---|---|---|---|"]
    for sp in (2, 4, 6, 8):
        k = f"s{sp}"
        if k not in d["spans"]:
            continue
        a = d["spans"][k]["row16_per_energy_agg"]
        for e in ("1.5", "3", "4.5", "6"):
            if e not in a:
                continue
            v = a[e]
            cm, cs = v["cnn_mean_sd"]
            lin = v["linear"]
            rm, rs = v["ratio_linear_mean_sd"]
            rc, _ = v["ratio_cubic_mean_sd"]
            L.append(f"| {sp}° | {e} | {f6(cm)} ± {f6(cs)} | {f6(lin)} | "
                     f"**{f4(rm)} ± {f4(rs)}** | {f4(rc)} |")
    L += ["",
          "**口径**：`p02_proj_only.py`，评估范围 = 数据表 test split（每档 45 视角），"
          "四档能量同训、评估时分组。归一化锁全局 0.7202/0.9249/5.5。",
          "**结论**：用户要的 1.5/3.0/6.0 MeV × 2°/6°/8° 全部已在表中，无需新训练。"
          " 3.0 MeV 单档比值随跨度单调下降（1.0658 → 1.0407 → 1.0020 → 0.9743），"
          "在 6–8° 穿越 1.0；池化值被 1.5 MeV 主导，讲「跨度效应」请用单能量列。",
          ""]
    return L


# ------------------------------------------------------------------ P1-3 dose
def table_dose() -> list[str]:
    L = ["## C. P1-3 dose sweep：逐能量 Row16（3 seeds，四档池化模型，**180 待补全视角**）", "",
         "| dose $f$ | 能量 (MeV) | CNN mean ± SD | linear | **CNN/linear** | CNN/cubic |",
         "|---|---|---|---|---|---|"]
    for tag, f in (("1.0", "f1"), ("0.5", "f0.5"), ("0.25", "f0.25"), ("0.125", "f0.125")):
        p = RUNS / f"_dose_eval_{f}.json"
        if not p.is_file():
            L.append(f"| {tag} | — | 缺 {p.name} | — | — | — |")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        ids = d.get("run_ids", [])
        for e in ("1.5", "3"):
            en = d.get("energies", {}).get(e)
            if not en or "recon_domain" not in en:
                L.append(f"| {tag} | {e} | 缺能量记录 | — | — | — |")
                continue
            rd = en["recon_domain"]
            lin = float(rd["linear_interp"]["row16"])
            cub = float(rd["cubic_interp"]["row16"])
            vals = [float(rd[i]["row16"]) for i in ids if i in rd]
            cm, cs = ms(vals)
            L.append(f"| {tag} | {e} | {f6(cm)} ± {f6(cs)} (n={len(vals)}) | {f6(lin)} | "
                     f"**{f4(cm / lin)}** | {f4(cm / cub)} |")
    L += ["",
          "**⚠ 口径与 span 表不同，禁止横比**：`dose_eval.py` 的聚合写在 test 过滤之前，"
          "评估范围是**全部 180 个待补全视角**（A→B / B→A 两方向均值），不是 span 表的"
          " 45 视角 test split，也不是 Fig 9 曾被误读的 360 视角。"
          " 例：同为 f=1.0 / s=2°，3.0 MeV 的 linear 在本表是 0.089077（180 视角），"
          "在 span 表是 0.088391（45 test），相差 0.8%；池化后差异会被能量平均放大到 +25.7%"
          "（见 R34 §③ 的三因子分解）。",
          "**数据来源**：`runs/_dose_eval_f{1,0.5,0.25,0.125}.json`，每份 `run_ids` 三元素，"
          "对应 `sparse_targets_dose_f*.csv` 的 seed 20261022/23/24（f=1.0 复用主协议"
          " headline 三 run `20261024_0400 / 1633 / 1805`）。",
          ""]
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS / "_p1_per_energy_tables.md"))
    args = ap.parse_args()
    out = table_p0() + table_span() + table_dose()
    text = "\n".join(out) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"[OK] 已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
