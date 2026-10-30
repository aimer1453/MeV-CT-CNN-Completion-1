#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/r34_report.py —— ① grouped split 实验的汇总与三套划分对照

回答的问题
----------
审稿意见：主结果的 100-epoch 版本用随机 split，但物模四重旋转对称会让 90° 旋转副本
泄漏进测试集；grouped split 只做过 30 epoch。→ 补 100-epoch grouped split。

本脚本把三套划分放在**同一张表**上：

  random      train 270 / val 45 / test 45   （原主协议，种子随机打乱）
  grouped     train  90 / val 90 / test 180  （扇区：0–90 / 90–180 / 180–360）
  contig270   train 270 / val 45 / test 45   （等样本量对照：0–270 / 270–315 / 315–360）

**contig270 是关键对照**：它的样本数与 random 完全相同、训练步数相同，
唯一差异是"训练角连续 vs 随机打乱"。因此：
  · 若 contig270 也崩 → 崩的原因是**角度分布不覆盖**（外推），不是样本量；
  · 若 contig270 正常    → 崩的原因是**样本量不足**（90 vs 270），分组本身无罪。

口径
----
  1D（cnn1d，C 组）：rmse_valid 是 Row 16 单行、mask=I>0&E>0；Row 16 coverage=1.0
                     故 rmse_valid ≡ rmse_full。
  2D（cnn，D 组）  ：rmse_valid 是 all 32 行 per-pixel masked。
  基线（linear / cubic）：由 r34_grouped_eval.py --baselines-only 在同一 split 上重算。

用法
----
    python scripts/r34_report.py
输出
----
    runs/_r34_summary.json + 控制台表
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

SPLITS = [
    ("random", "csv_data/sparse_targets.csv", "_r34_rand_baselines.json",
     ["sparse_targets.csv", "sparse_targets_3MeV.csv"]),
    ("grouped", "csv_data/sparse_targets_3MeV_grouped.csv", "_r34_grp_baselines.json",
     ["sparse_targets_3MeV_grouped.csv"]),
    ("contig270", "csv_data/sparse_targets_3MeV_contig270.csv",
     "_r34_contig270_baselines.json", ["sparse_targets_3MeV_contig270.csv"]),
    # near45：单变量隔离 —— 训练量**与 grouped 相同**（90 行），但 test 紧邻训练末端
    # （90–135°，距 train 末角仅 1°）。用于把"训练量不足"和"远端外推"两个变量拆开：
    #   near45 崩 → 崩因是训练量（45 视角太少）
    #   near45 正常 → 崩因是远端外推（test 距 train 90–270°）
    ("near45", "csv_data/sparse_targets_3MeV_near45.csv",
     "_r34_near45_baselines.json", ["sparse_targets_3MeV_near45.csv"]),
    # rand90：任务 2 的控制 —— 训练行数与 near45 相同（90）、test 角**逐位相同**
    # （90–134°），唯一差异是训练角从"扇区 0–89°"改成"全圆随机 90 个"。
    #   rand90 ≈ random → 崩因是角度覆盖；rand90 ≈ near45 → 崩因是训练量
    ("rand90", "csv_data/sparse_targets_3MeV_rand90.csv",
     "_r34_rand90_baselines.json", ["sparse_targets_3MeV_rand90.csv"]),
    # even90：任务 2 加强版 —— 同 90 行、同 test 块，但训练角在非 test 角上等间隔
    # 铺满全圆（中位间隙 4°，最大间隙 46° 即 test 块本身），test 块两端有 89/135 夹住。
    # 用于把"全局密度/条件覆盖"这一项做到最有利，看训练量本身还剩多少退化。
    ("even90", "csv_data/sparse_targets_3MeV_even90.csv",
     "_r34_even90_baselines.json", ["sparse_targets_3MeV_even90.csv"]),
]
#: 别名说明：七种子侦察用 sparse_targets_3MeV.csv，其 3.0 MeV 的 test 角集合与
#: sparse_targets.csv 逐位相同（已核，45 个），故归入 random 一支。
#:
#: random 的 2D 一支必须限定白名单：历史上所有 run 都用 sparse_targets.csv，
#: 直接平均会把 30+ 个不同用途的 run（消融、调参、旧协议）混进来。这里只取
#: Table 5 的 headline 三 seed（与 §5.6 的三种子统计同一批）。
RANDOM_2D_WHITELIST = [
    "20261024_0400_cnn_bs8_lr1e-4",
    "20261024_1633_cnn_bs8_lr1e-4",
    "20261024_1805_cnn_bs8_lr1e-4",
]

#: 2D 架构（cnn，四档能量一起训练）的 `test_metrics.overall` 是 **四档池化**值，
#: 而 1D 架构（cnn1d，3.0 MeV 单档）是单档值。两者直接对着同一个 "linear_row16"
#: 比会**跨口径**：早先 R34 表里 "2D all32 +52.2%" 就是用池化 CNN(0.3805) 比
#: 3.0 MeV linear(0.7960) 得出的，偏高。这里改从 `r34_grouped_eval.py --cnn-run`
#: 的 3.0 MeV 单档评估里取 all32 / row16 / 图像域，与 1D 支同能量同 mask。
EVAL_2D_GLOB = {
    "random": ["_r34_rand2d_s*.json"],
    "grouped": ["_r34_grp2d_s*.json"],
}


def csv_test_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    out: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "test":
                out[r["energy_MeV"]] = out.get(r["energy_MeV"], 0) + 1
    return out


def collect_runs() -> list[dict]:
    """扫描 runs/，只收 3.0 MeV 相关的、有 test_metrics 的 run。"""
    out = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        cfg_p, met_p = d / "config.json", d / "test_metrics.json"
        if not (cfg_p.is_file() and met_p.is_file()):
            continue
        try:
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            met = json.loads(met_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        data = cfg.get("data", {})
        tgt = str(data.get("targets") or data.get("csv") or "")
        # 只看 3 MeV 单能量或含 3 MeV 的 4 能量基线集合
        energies = data.get("energies_MeV") or []
        if not tgt:
            continue
        arch = "1d" if met.get("arch") == "UNet1D" else "2d"
        if arch == "2d" and energies and not any(abs(float(e) - 3.0) < 1e-9 for e in energies):
            continue
        seed_v = cfg.get("seed")
        if seed_v is None:
            continue                      # 老 run 无 seed 字段，跳过
        out.append(dict(
            run_id=d.name, arch=arch, csv=Path(tgt).name,
            seed=int(seed_v), epoch=met.get("best_epoch"),
            rmse_valid=float(met["overall"]["rmse_valid"]),
            rmse_full=float(met["overall"]["rmse_full"]),
            n_samples=int(met["overall"]["n_samples"]),
        ))
    return out


def main() -> int:
    runs = collect_runs()
    baselines = {}
    for name, csv_rel, json_rel, _alias in SPLITS:
        p = RUNS / json_rel
        if p.is_file():
            baselines[name] = json.loads(p.read_text(encoding="utf-8"))

    summary: dict = {"splits": {}, "runs": runs}

    for name, csv_rel, _j, aliases in SPLITS:
        csv_p = ROOT / csv_rel
        tc = csv_test_counts(csv_p)
        entry = {"csv": csv_rel, "test_views_by_energy": tc, "arms": {}}
        b = baselines.get(name)
        if b:
            e = "3"
            entry["baseline"] = {
                "linear_row16": b["projection"]["linear"]["row16"]["overall"]["rmse_valid"],
                "cubic_row16": b["projection"]["cubic"]["row16"]["overall"]["rmse_valid"],
                "linear_all32": b["projection"]["linear"]["all32"]["overall"]["rmse_valid"],
                "cubic_all32": b["projection"]["cubic"]["all32"]["overall"]["rmse_valid"],
                "n_test_views": b["n_test_views"],
                "image_linear": b["image"]["linear"]["overall"]["rmse_valid"],
                "image_cubic": b["image"]["cubic"]["overall"]["rmse_valid"],
            }
        for arch in ("1d", "2d"):
            sub = [r for r in runs if r["csv"] in aliases and r["arch"] == arch]
            if name == "random" and arch == "2d":
                sub = [r for r in sub if r["run_id"] in RANDOM_2D_WHITELIST]
            if not sub:
                continue
            vals = np.array([r["rmse_valid"] for r in sub], dtype=float)
            entry["arms"][arch] = {
                "n_seeds": len(sub), "seeds": [r["seed"] for r in sub],
                "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "runs": [r["run_id"] for r in sub],
                "energy_MeV": "pooled(1.5/3/4.5/6)" if arch == "2d" else 3.0,
                "metric": "all32" if arch == "2d" else "row16",
                "source": "test_metrics.overall（四档池化）" if arch == "2d" else "test_metrics.overall",
            }

        # 2D 支：用 3.0 MeV 单档评估覆盖上面的池化值（口径与 1D 支对齐）
        jsons = []
        for pat in EVAL_2D_GLOB.get(name, []):
            jsons += sorted((RUNS).glob(pat))
        if jsons:
            r16, a32, img, img_lin, imgl_seeds = [], [], [], [], []
            for j in jsons:
                d = json.loads(j.read_text(encoding="utf-8"))
                r16.append(d["projection"]["cnn"]["row16"]["overall"]["rmse_valid"])
                a32.append(d["projection"]["cnn"]["all32"]["overall"]["rmse_valid"])
                img.append(d["image"]["cnn"]["overall"]["rmse_valid"])
                img_lin.append(d["image"]["linear"]["overall"]["rmse_valid"])
                imgl_seeds.append(Path(j).name)
            entry["arms"]["2d"] = {
                "n_seeds": len(a32), "mean_all32_3MeV": float(np.mean(a32)),
                "sd_all32_3MeV": float(np.std(a32, ddof=1)) if len(a32) > 1 else 0.0,
                "mean_row16_3MeV": float(np.mean(r16)),
                "sd_row16_3MeV": float(np.std(r16, ddof=1)) if len(r16) > 1 else 0.0,
                "image_cnn_3MeV": float(np.mean(img)),
                "image_linear_3MeV": float(np.mean(img_lin)),
                "energy_MeV": 3.0, "metric": "all32",
                "source": "r34_grouped_eval.py --cnn-run（3.0 MeV 单档）",
                "eval_files": imgl_seeds,
                # 保留池化值以便溯源，但**不要**拿它跟 3.0 MeV 的 linear 比
                "pooled_test_metrics_mean": entry["arms"]["2d"].get("mean"),
            }
        summary["splits"][name] = entry

    # ---------------- 控制台报表 ----------------
    print("=" * 96)
    print("① 100-epoch grouped split 实验：三套划分对照（3.0 MeV）")
    print("=" * 96)
    for name, csv_rel, _j, _alias in SPLITS:
        e = summary["splits"][name]
        bl = e.get("baseline", {})
        print(f"\n■ {name:10s}  {csv_rel}")
        print(f"   test 视角数: {e['test_views_by_energy']}")
        if bl:
            print(f"   基线 row16 : linear={bl['linear_row16']:.6f}  cubic={bl['cubic_row16']:.6f}")
            print(f"   基线 all32 : linear={bl['linear_all32']:.6f}  cubic={bl['cubic_all32']:.6f}")
        for arch, lab in (("1d", "1D Row16 (cnn1d)"), ("2d", "2D all32 (cnn)")):
            a = e["arms"].get(arch)
            if not a:
                print(f"   {lab:18s} —— 无 run")
                continue
            if arch == "1d":
                mean_, sd_ = a["mean"], a["sd"]
                base = bl.get("linear_row16")
            else:
                mean_, sd_ = a["mean_all32_3MeV"], a["sd_all32_3MeV"]
                base = bl.get("linear_all32")
            rel = (base - mean_) / base * 100 if base else float("nan")
            print(f"   {lab:18s} n={a['n_seeds']}  {mean_:.6f} ± {sd_:.6f}"
                  f"   vs linear: {rel:+.1f}%   [{a['energy_MeV']} MeV]")
            if arch == "2d":
                print(f"   {'':18s} 池化四档(test_metrics) {a['pooled_test_metrics_mean']:.6f}"
                      f"  ← 仅供溯源，勿与 3.0 MeV linear 比")

    # ---------------- 判定 ----------------
    print("\n" + "=" * 96)
    print("判定要点")
    print("=" * 96)
    grp = summary["splits"]["grouped"]["arms"]
    ran = summary["splits"]["random"]["arms"]
    con = summary["splits"]["contig270"]["arms"]
    near = summary["splits"].get("near45", {}).get("arms", {})
    g1, r1, c1, n1 = grp.get("1d"), ran.get("1d"), con.get("1d"), near.get("1d")

    print("  2x2 设计（训练量 × test 距训练末端的距离），1D Row16：")
    print("  %-28s %-14s %-16s %s" % ("配置", "训练行数", "test 位置", "RMSE"))
    if r1:
        print("  %-28s %-14s %-16s %.6f" % ("random", "270", "全圆内插", r1["mean"]))
    if c1:
        print("  %-28s %-14s %-16s %.6f" % ("contig270", "270", "315-360（45° 外推）", c1["mean"]))
    if n1:
        print("  %-28s %-14s %-16s %.6f" % ("near45", "90", "90-135（紧邻外推）", n1["mean"]))
    if g1:
        print("  %-28s %-14s %-16s %.6f" % ("grouped", "90", "180-360（远距外推）", g1["mean"]))

    print()
    if g1 and r1:
        print(f"  · grouped vs random（1D Row16）：{g1['mean']:.6f} vs {r1['mean']:.6f}"
              f"  → ×{g1['mean']/r1['mean']:.2f}")
    if c1 and r1:
        print(f"  · contig270 vs random：{c1['mean']:.6f} vs {r1['mean']:.6f}"
              f"  → ×{c1['mean']/r1['mean']:.2f}（训练量与 random 相同、test 在外推区，仍正常）")
    if n1 and r1 and g1 and c1:
        print(f"  · near45 vs random 差 {n1['mean']-r1['mean']:+.6f}；"
              f"vs grouped 差 {n1['mean']-g1['mean']:+.6f}")
        print()
        print("  【定量分解】把两个变量各自的影响拆开（相对 random = 1.00）：")
        print("    ① 只改外推距离（训练量固定 270）：random → contig270 = "
              f"×{c1['mean']/r1['mean']:.2f}")
        print("    ② 只改训练量（test 固定在紧邻外推区）：contig270 → near45 = "
              f"×{n1['mean']/c1['mean']:.2f}")
        print("    ③ 两者同时变：random → grouped = "
              f"×{g1['mean']/r1['mean']:.2f}")
        print()
        print("    ⇒ 训练量 270→90 的退化贡献 ×%.2f（+%.0f%%）"
              % (n1["mean"] / c1["mean"], (n1["mean"] / c1["mean"] - 1) * 100))
        print("    ⇒ 外推距离 45°→90–270° 的退化贡献 ×%.2f（+%.0f%%）"
              % (g1["mean"] / n1["mean"], (g1["mean"] / n1["mean"] - 1) * 100))
        print("    ⇒ **两个因素叠加**才造成 ×%.2f 的崩溃，主因是远端外推，"
              "但训练量不足本身已足以让 CNN 输给线性插值：" % (g1["mean"] / r1["mean"]))
        bl_near = summary["splits"]["near45"].get("baseline", {})
        if bl_near:
            print("       near45 = %.6f vs linear %.6f → CNN **输 %.1f%%**"
                  % (n1["mean"], bl_near["linear_row16"],
                     (n1["mean"] / bl_near["linear_row16"] - 1) * 100))

    p = RUNS / "_r34_summary.json"
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
