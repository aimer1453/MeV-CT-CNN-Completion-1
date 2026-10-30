#!/usr/bin/env python
"""Row-16-only 消融评估：三方对比（3.0 MeV，Row 16 口径）。

对比对象
--------
1. **原文跨能量 32 行**（已发布基线）：4 能量 × 270 = 1080 样本训练，
   取自 `runs/_dose_eval_f1.json` 中 3.0 MeV 的 row16（3 种子均值 0.095155）。
2. **A 组 单能量 32 行**（本次对照）：3.0 MeV 270 样本，输入正常 32 行。
3. **B 组 单能量 Row-16-only**（本次消融）：3.0 MeV 270 样本，输入只保留 Row 16。

A/B 两组唯一差别是「另外 31 行有没有真实信息」——网络结构、参数量、
batch、lr、epoch、seed 全部相同，因此 A vs B 是干净的归因。
1 vs A 的差别是训练数据规模，用于说明"跨能量基线为什么不能直接当对照"。

评估口径
--------
复用 `dose_eval.py`（--factor 1.0 + 单能量表），取 3.0 MeV 的 row16 RMSE。
linear/cubic 是确定性插值，与训练无关，三组共用同一数值，可直接横比。

用法
----
    python scripts/eval_row16_ablation.py
产出：runs/_row16_ablation_eval.json 、 runs/_row16_ablation_eval.md
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
RUNS = ROOT / "runs"
INDEX = RUNS / "_row16_ablation_index.csv"
F1_EVAL = RUNS / "_dose_eval_f1.json"       # 原文跨能量 32 行基线
PY = sys.executable or r"<user-python>"

ENERGY_KEY = "3"                             # 3.0 MeV
CSV_NAME = resolve_csv("sparse_targets_3MeV.csv")


def group_run_ids(group: str) -> list[str]:
    if not INDEX.is_file():
        raise SystemExit(f"[FATAL] 缺 {INDEX}：先跑 scripts/row16_ablation.py")
    out = [r["run_id"] for r in csv.DictReader(open(INDEX, newline="", encoding="utf-8"))
           if r["group"] == group]
    if len(out) != 3:
        raise SystemExit(f"[FATAL] {group} 只有 {len(out)} 个 run，期望 3")
    return out


def run_eval(group: str, run_ids: list[str], single_row: bool = False) -> dict:
    out = RUNS / f"_eval_row16_{group}.json"
    cmd = [PY, str(ROOT / "scripts" / "dose_eval.py"), "--factor", "1.0",
           "--csv", CSV_NAME, "--config", "config.json",
           "--run-ids", *run_ids, "--out", str(out)]
    if single_row:
        cmd.append("--single-row")
    print(f"[RUN] {group} (single_row={single_row}): {' '.join(cmd[-4:])}", flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc != 0:
        raise SystemExit(f"[FATAL] dose_eval {group} rc={rc}")
    return json.loads(out.read_text(encoding="utf-8"))


def row16_of(d: dict, method: str) -> float | None:
    e = d.get("energies", {}).get(ENERGY_KEY)
    if not e:
        return None
    return float(e["recon_domain"][method]["row16"])


def summarize_group(d: dict, run_ids: list[str]) -> dict:
    cnn = [row16_of(d, r) for r in run_ids]
    cnn = [v for v in cnn if v is not None]
    lin = row16_of(d, "linear_interp")
    cub = row16_of(d, "cubic_interp")
    if not cnn or lin is None:
        raise SystemExit("[FATAL] 评测 JSON 里缺 3.0 MeV 的 row16，检查数据表是否只有 3.0 MeV")
    m = float(np.mean(cnn))
    return {
        "cnn_per_seed": [round(float(v), 6) for v in cnn],
        "cnn_mean": m,
        "cnn_std": float(np.std(cnn, ddof=1)) if len(cnn) > 1 else 0.0,
        "linear": float(lin),
        "cubic": float(cub),
        "imp_vs_linear_pct": (float(lin) - m) / float(lin) * 100.0,
        "imp_vs_cubic_pct": ((float(cub) - m) / float(cub) * 100.0) if cub else None,
    }


def main() -> int:
    out = {"energy_MeV": 3.0, "metric": "row16 RMSE (per-pixel mask, -log 物理域)"}

    # ---- 1. 原文跨能量 32 行基线（不重跑，直接取已发布评测） ----
    if not F1_EVAL.is_file():
        raise SystemExit(f"[FATAL] 缺 {F1_EVAL}")
    f1 = json.loads(F1_EVAL.read_text(encoding="utf-8"))
    base_ids = f1["run_ids"]
    cnn_f1 = [row16_of(f1, r) for r in base_ids]
    lin_f1 = row16_of(f1, "linear_interp")
    m_f1 = float(np.mean([v for v in cnn_f1 if v is not None]))
    out["ref_cross_energy_32row"] = {
        "note": "4 能量 × 270 = 1080 样本训练（论文正文配置），非本消融的训练条件",
        "run_ids": base_ids,
        "cnn_per_seed": [round(float(v), 6) for v in cnn_f1 if v is not None],
        "cnn_mean": m_f1,
        "linear": float(lin_f1),
        "imp_vs_linear_pct": (float(lin_f1) - m_f1) / float(lin_f1) * 100.0,
    }

    # ---- 2. A / B 两组 ----
    for group, label, single_row in (("base32_3mev", "A_单能量32行", False),
                                     ("row16_3mev", "B_单能量Row16only", True)):
        rids = group_run_ids(group)
        d = run_eval(group, rids, single_row=single_row)
        s = summarize_group(d, rids)
        s["run_ids"] = rids
        out[label] = s

    # ---- 3. 归因判定 ----
    a, b = out["A_单能量32行"], out["B_单能量Row16only"]
    d_imp = b["imp_vs_linear_pct"] - a["imp_vs_linear_pct"]
    d_abs = b["cnn_mean"] - a["cnn_mean"]
    rel = d_abs / a["cnn_mean"] * 100.0
    out["attribution"] = {
        "delta_improvement_pp": d_imp,
        "delta_row16_rmse": d_abs,
        "relative_change_pct": rel,
        "verdict": None,
    }
    if abs(rel) < 5.0:
        out["attribution"]["verdict"] = (
            "B 与 A 相差 <5%：Row-16-only 训练与 32 行训练在该行上几乎等价，"
            "可放心改为 Row 16-only（跨行冗余不是 CNN 优势的关键来源）")
    elif rel > 0:
        out["attribution"]["verdict"] = (
            f"B 比 A 的 Row 16 RMSE 恶化 {rel:.1f}%：抹掉另外 31 行会显著伤害该行预测，"
            "说明**跨行冗余确实是 CNN 优势的关键来源**（支持保留 32 行训练）")
    else:
        out["attribution"]["verdict"] = (
            f"B 反而比 A 好 {-rel:.1f}%：单行训练在该行上更准，可考虑改为 Row 16-only")

    jp = RUNS / "_row16_ablation_eval.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] {jp}")
    write_md(out, RUNS / "_row16_ablation_eval.md")
    print(f"[md]   {RUNS / '_row16_ablation_eval.md'}")

    print("\n=== 三方对比（3.0 MeV，Row 16 RMSE）===")
    for key in ("ref_cross_energy_32row", "A_单能量32行", "B_单能量Row16only"):
        g = out[key]
        print(f"  {key:24s} CNN={g['cnn_mean']:.6f}  linear={g['linear']:.6f}  "
              f"imp={g['imp_vs_linear_pct']:+.2f}%")
    print(f"  判定: {out['attribution']['verdict']}")
    return 0


def write_md(out: dict, path: Path) -> None:
    L = ["# Row-16-only 消融（3.0 MeV，Row 16 口径）\n",
         "> 指标：Row 16 的 per-pixel mask RMSE（-log 物理域）。"
         "improvement% = (linear − CNN)/linear，**正 = CNN 更好**。\n",
         "\n## 三方对比\n",
         "| 组 | 训练数据 | 输入行数 | CNN row16 (3 种子均值) | linear row16 | improvement % |"
         " 参数量 |",
         "|---|---|---|---|---|---|---|"]
    rows = [
        ("原文跨能量 32 行", "4 能量 × 270 = 1080", "32", out["ref_cross_energy_32row"]),
        ("A 单能量 32 行", "3.0 MeV × 270", "32", out["A_单能量32行"]),
        ("B 单能量 Row-16-only", "3.0 MeV × 270", "1（其余填零计数填充值）",
         out["B_单能量Row16only"]),
    ]
    for name, data_, inrows, g in rows:
        L.append(f"| {name} | {data_} | {inrows} | {g['cnn_mean']:.6f} "
                 f"(±{g.get('cnn_std', 0):.4f}) | {g['linear']:.6f} | "
                 f"{g['imp_vs_linear_pct']:+.2f} | 1,950,017 |")
    a = out["attribution"]
    L.append(f"\n## 归因\n")
    L.append(f"- A vs B 的 Row 16 RMSE 差 = {a['delta_row16_rmse']:+.6f} "
             f"（相对 {a['relative_change_pct']:+.1f}%），"
             f"improvement 差 = {a['delta_improvement_pp']:+.2f} 个百分点。\n")
    L.append(f"- **判定**：{a['verdict']}\n")
    L.append("\n## 口径说明\n")
    L.append("- A/B 两组：网络结构、参数量（1,950,017）、batch=8、lr=1e-4、epoch=100、"
             "seed 集合完全相同；唯一差别是另外 31 行是否为真实数据。\n")
    L.append("- 「原文跨能量 32 行」是 1080 样本训练，与 A 组的差别是**训练数据规模**，"
             "所以它不能替代 A 作为对照，仅作参考。\n")
    L.append("- linear/cubic 是确定性插值，与训练无关，三组共用同一数值，可横比。\n")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
