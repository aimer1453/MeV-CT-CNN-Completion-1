#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/r34_span_table.py —— 任务 1：跨度扫描的**三种子**汇总表

背景
----
论文 §5.6 / Table 7 的跨度比值是**单种子**点值（每个跨度只有一个 CNN run），
主文已注明"单种子探索性、不建立跨度排序"。本脚本给每个跨度补到 **3 种子**，
给出 CNN/linear 与 CNN/cubic 的 **mean ± SD**，并回答"6°/8° 处 CNN 是否仍领先"。

run 归属（重要，config 字段有坑）
--------------------------------
· 2026-10-24 那一批（种子 20261022）四个 run 的 `config.data.targets` **都写成
  `sparse_targets.csv`**（写入逻辑在 --csv 覆盖时未记实际表名，后来的 run 已修）。
  实际用的是跨度表：证据是 `runs/_reviewer_p0/logs/e100_s{2,4,6,8}.log` 首行
  `[CSV] 使用 --csv 覆盖数据表: sparse_targets_sN.csv`。故这里**按日志登记**，
  不按 config 自动发现。
· 新补的种子 run（20261029_*_stepN_sXX）config 已正确记录 `sparse_targets_sN.csv`，
  按 config 自动发现。
· s=2 的几何与主协议（`sparse_targets.csv`，step=2/h=1）**完全相同**，故 s=2 的
  种子 20261023 / 20261024 直接复用主协议 headline 2D 三 seed 中的另外两个
  （`20261024_1633` / `20261024_1805`），不需要重训。已核：三者同 bs/lr/100ep/
  同 5 通道/同归一化。

口径
----
与 `p02_proj_only.py` 一致（同 mask、同 test 角、先逐能量池化再跨能量）：
  · pooled row16  = 4 档能量先各自算 MSE，再取平均后开方（论文 Table 7 的 Overall 列）
  · 单能量 row16  = 该能量自己的 masked RMSE（论文 Table 7 的 1.5/3.0/4.5/6.0 列）
  · ratio         = CNN / linear（<1 表示 CNN 更好）；cubic 同理

用法
----
    python scripts/r34_span_table.py            # 缺哪个 run 的评估就自动补跑
    python scripts/r34_span_table.py --no-eval  # 只读已有 json
输出
----
    runs/_r34_span_sweep.json  + 控制台 Markdown 表
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT_DIR = RUNS / "_reviewer_p0"
PY = str(ROOT / "venv" / "Scripts" / "python.exe")
SPANS = [2, 4, 6, 8]
ENERGIES = ["1.5", "3", "4.5", "6"]

#: 2026-10-24 单种子批次：config 的 targets 字段过期，按日志登记（见文件头）
LEGACY = {
    (2, 20261022): "20261024_0400_cnn_bs8_lr1e-4",
    (4, 20261022): "20261024_0515_cnn_bs8_lr1e-4",
    (6, 20261022): "20261024_0550_cnn_bs8_lr1e-4",
    (8, 20261022): "20261024_0625_cnn_bs8_lr1e-4",
}
#: s=2 == 主协议几何，复用主协议 headline 2D 三 seed 的另两个
S2_MAIN_CSV = {
    20261023: "20261024_1633_cnn_bs8_lr1e-4",
    20261024: "20261024_1805_cnn_bs8_lr1e-4",
}


def discover_from_config() -> dict[tuple[int, int], str]:
    """新 run：config.data.targets == sparse_targets_s<s>.csv，且有 test_metrics.json。"""
    out: dict[tuple[int, int], str] = {}
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        cfg_p, met_p = d / "config.json", d / "test_metrics.json"
        if not (cfg_p.is_file() and met_p.is_file()):
            continue
        try:
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        tgt = str(cfg.get("data", {}).get("targets") or "")
        for s in SPANS:
            if tgt.endswith(f"sparse_targets_s{s}.csv"):
                seed = int(cfg["seed"])
                out[(s, seed)] = d.name
    return out


def eval_json(span: int, run_id: str) -> Path:
    return OUT_DIR / f"projonly_s{span}_{run_id}.json"


def ensure_eval(span: int, run_ids: list[str], do_eval: bool) -> None:
    missing = [r for r in run_ids if not eval_json(span, r).is_file()]
    if not missing:
        return
    if not do_eval:
        raise SystemExit(f"[FATAL] 缺评估 (s={span}): {missing}，去掉 --no-eval 重跑")
    cmd = [PY, "scripts/p02_proj_only.py", "--step", str(span)]
    for r in missing:
        cmd += ["--cnn-run", r]
    print(f"[eval] s={span}°  {len(missing)} 个 run: {missing}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def load(span: int, run_id: str) -> dict:
    return json.loads(eval_json(span, run_id).read_text(encoding="utf-8"))["projection"]


def ms(vals: list[float]) -> tuple[float, float]:
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    manifest: dict[tuple[int, int], str] = dict(LEGACY)
    manifest.update(discover_from_config())
    for seed, run_id in S2_MAIN_CSV.items():
        manifest.setdefault((2, seed), run_id)

    for s in SPANS:
        runs = [manifest[(s, seed)] for seed in sorted(k[1] for k in manifest if k[0] == s)]
        ensure_eval(s, runs, not args.no_eval)

    summary: dict = {"spans": {}, "manifest": {f"s{s}_seed{seed}": run
                                               for (s, seed), run in manifest.items()}}
    for s in SPANS:
        seeds = sorted(seed for (sp, seed) in manifest if sp == s)
        arms = {"row16": {}, "all32": {}, "row16_per_energy": {}}
        for seed in seeds:
            p = load(s, manifest[(s, seed)])
            arms["row16"][seed] = {"cnn": p["cnn"]["row16"]["overall"]["rmse_valid"],
                                   "linear": p["linear"]["row16"]["overall"]["rmse_valid"],
                                   "cubic": p["cubic"]["row16"]["overall"]["rmse_valid"]}
            arms["all32"][seed] = {"cnn": p["cnn"]["all32"]["overall"]["rmse_valid"],
                                   "linear": p["linear"]["all32"]["overall"]["rmse_valid"],
                                   "cubic": p["cubic"]["all32"]["overall"]["rmse_valid"]}
            arms["row16_per_energy"][seed] = {
                e: {"cnn": p["cnn"]["row16"]["per_energy"][e]["rmse_valid"],
                    "linear": p["linear"]["row16"]["per_energy"][e]["rmse_valid"],
                    "cubic": p["cubic"]["row16"]["per_energy"][e]["rmse_valid"]}
                for e in ENERGIES}
        agg = {}
        for roi in ("row16", "all32"):
            seed_vals = arms[roi]
            agg[roi] = {
                "n_seeds": len(seed_vals),
                "cnn_mean_sd": ms([v["cnn"] for v in seed_vals.values()]),
                "linear": ms([v["linear"] for v in seed_vals.values()]),
                "cubic_mean_sd": ms([v["cubic"] for v in seed_vals.values()]),
                "ratio_linear_mean_sd": ms([v["cnn"] / v["linear"] for v in seed_vals.values()]),
                "ratio_cubic_mean_sd": ms([v["cnn"] / v["cubic"] for v in seed_vals.values()]),
                "ratio_linear_per_seed": [v["cnn"] / v["linear"] for v in seed_vals.values()],
                "ratio_cubic_per_seed": [v["cnn"] / v["cubic"] for v in seed_vals.values()],
            }
        pe = {}
        for e in ENERGIES:
            pe[e] = {
                "ratio_linear_mean_sd": ms([arms["row16_per_energy"][sd][e]["cnn"]
                                            / arms["row16_per_energy"][sd][e]["linear"]
                                            for sd in seeds]),
                "ratio_cubic_mean_sd": ms([arms["row16_per_energy"][sd][e]["cnn"]
                                           / arms["row16_per_energy"][sd][e]["cubic"]
                                           for sd in seeds]),
                "cnn_mean_sd": ms([arms["row16_per_energy"][sd][e]["cnn"] for sd in seeds]),
                "linear": ms([arms["row16_per_energy"][sd][e]["linear"] for sd in seeds])[0],
            }
        summary["spans"][f"s{s}"] = {"seeds": seeds, "per_seed": arms, "agg": agg,
                                     "row16_per_energy_agg": pe}

    p = RUNS / "_r34_span_sweep.json"
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- 控制台表 ----------------
    print("\n" + "=" * 108)
    print("任务 1：跨度扫描 3 种子（100 ep，4 档能量池化；ratio<1 = CNN 优于基线）")
    print("=" * 108)
    print(f"{'span':>6} | {'n':>2} | {'CNN row16':>17} | {'linear row16':>13} | "
          f"{'CNN/lin':>14} | {'CNN/cub':>14} | {'CNN all32':>17} | {'all32 ratio':>13}")
    print("-" * 108)
    for s in SPANS:
        a = summary["spans"][f"s{s}"]["agg"]
        r, r32 = a["row16"], a["all32"]
        print(f"{s:>4}° | {r['n_seeds']:>2} | {r['cnn_mean_sd'][0]:.6f}±{r['cnn_mean_sd'][1]:.6f} | "
              f"{r['linear'][0]:.6f}      | {r['ratio_linear_mean_sd'][0]:.4f}±{r['ratio_linear_mean_sd'][1]:.4f} | "
              f"{r['ratio_cubic_mean_sd'][0]:.4f}±{r['ratio_cubic_mean_sd'][1]:.4f} | "
              f"{r32['cnn_mean_sd'][0]:.6f}±{r32['cnn_mean_sd'][1]:.6f} | "
              f"{r32['ratio_linear_mean_sd'][0]:.4f}±{r32['ratio_linear_mean_sd'][1]:.4f}")
    print("\n逐能量 Row16：CNN/linear（上）与 CNN/cubic（下）")
    print(f"{'span':>6} | " + " | ".join(f"{e+' MeV':>15}" for e in ENERGIES))
    print("-" * 108)
    for s in SPANS:
        pe = summary["spans"][f"s{s}"]["row16_per_energy_agg"]
        print(f"{s:>4}° | " + " | ".join(
            f"{pe[e]['ratio_linear_mean_sd'][0]:.4f}±{pe[e]['ratio_linear_mean_sd'][1]:.4f}" for e in ENERGIES))
        print(f"{'':>6} | " + " | ".join(
            f"{pe[e]['ratio_cubic_mean_sd'][0]:.4f}±{pe[e]['ratio_cubic_mean_sd'][1]:.4f}" for e in ENERGIES))
    print(f"\n[json] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
