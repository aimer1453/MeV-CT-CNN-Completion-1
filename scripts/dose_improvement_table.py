#!/usr/bin/env python
"""P0-1 · CNN-vs-linear improvement% 表（Row 16 + all32，跨能量等权池化）。

improvement% = (linear − CNN) / linear × 100；正 = CNN 更好，负 = CNN 更差。
池化口径：先对每能量取 CNN 三种子均值，再对 4 能量取等权算术平均（与 §5.6 bootstrap 一致）。

输出：docs/figs/p01_dose_rmse/improvement_table.md  +  runs/_dose_improvement.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "docs/figs/p01_dose_rmse"
FACTORS = [1.0, 0.5, 0.25, 0.125]
ENERGIES = [1.5, 3.0, 4.5, 6.0]


def load(f: float) -> dict:
    p = RUNS / f"_dose_eval_f{f:g}.json"
    if not p.is_file():
        raise SystemExit(f"[FATAL] 缺 {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def cnn_ids(d: dict) -> list[str]:
    return [k for k in d["energies"]["1.5"]["recon_domain"].keys()
            if k not in ("linear_interp", "cubic_interp")]


def main() -> int:
    rows = []
    per_e: dict = {}
    for f in FACTORS:
        d = load(f)
        rids = cnn_ids(d)
        acc = {"c_r16": [], "l_r16": [], "c_a32": [], "l_a32": []}
        pe = {}
        for e in ENERGIES:
            ek = f"{e:g}"
            rd = d["energies"][ek]["recon_domain"]
            cnn_r16 = float(np.mean([rd[r]["row16"] for r in rids]))
            lin_r16 = float(rd["linear_interp"]["row16"])
            cnn_a32 = float(np.mean([rd[r]["all32"] for r in rids]))
            lin_a32 = float(rd["linear_interp"]["all32"])
            acc["c_r16"].append(cnn_r16); acc["l_r16"].append(lin_r16)
            acc["c_a32"].append(cnn_a32); acc["l_a32"].append(lin_a32)
            pe[ek] = dict(cnn_r16=cnn_r16, lin_r16=lin_r16,
                          cnn_a32=cnn_a32, lin_a32=lin_a32)
        c_r16m = float(np.mean(acc["c_r16"])); l_r16m = float(np.mean(acc["l_r16"]))
        c_a32m = float(np.mean(acc["c_a32"])); l_a32m = float(np.mean(acc["l_a32"]))
        imp_r = (l_r16m - c_r16m) / l_r16m * 100.0
        imp_a = (l_a32m - c_a32m) / l_a32m * 100.0
        rows.append(dict(f=f, cnn_row16=c_r16m, linear_row16=l_r16m, imp_row16=imp_r,
                         cnn_all32=c_a32m, linear_all32=l_a32m, imp_all32=imp_a))
        per_e[f"{f:g}"] = pe

    md = ["# P0-1 · CNN vs linear improvement %", "",
          "> improvement% = (linear − CNN) / linear × 100；**正 = CNN 更好，负 = CNN 更差**。",
          "> 池化：每能量取 CNN 三种子均值，再对 4 能量等权算术平均（与 §5.6 bootstrap 同口径）。",
          "",
          "| f | CNN row16 | linear row16 | improvement % (row16) | CNN all32 | linear all32 | improvement % (all32) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['f']:g} | {r['cnn_row16']:.5f} | {r['linear_row16']:.5f} | "
                  f"{r['imp_row16']:+.2f} | {r['cnn_all32']:.5f} | {r['linear_all32']:.5f} | "
                  f"{r['imp_all32']:+.2f} |")
    md += ["", "## 逐能量明细", ""]
    for f in FACTORS:
        md.append(f"### f = {f:g}")
        md.append("| energy | CNN r16 | lin r16 | imp% r16 | CNN a32 | lin a32 | imp% a32 |")
        md.append("|---|---|---|---|---|---|---|")
        for e in ENERGIES:
            pe = per_e[f"{f:g}"][f"{e:g}"]
            ir = (pe["lin_r16"] - pe["cnn_r16"]) / pe["lin_r16"] * 100.0
            ia = (pe["lin_a32"] - pe["cnn_a32"]) / pe["lin_a32"] * 100.0
            md.append(f"| {e:g} | {pe['cnn_r16']:.5f} | {pe['lin_r16']:.5f} | {ir:+.2f} | "
                      f"{pe['cnn_a32']:.5f} | {pe['lin_a32']:.5f} | {ia:+.2f} |")
        md.append("")
    md += ["## 解读",
           "- **Row 16（重建所用行）**：高剂量下 CNN 与线性插值基本打平甚至略差"
           "（1.5 MeV 例外，CNN 领先）；这是探测器峰顶计数充足、线性插值已很准所致。",
           "- **all 32 行（探测器全域）**：CNN 在所有剂量档**决定性占优**"
           "（improvement 大正），且剂量越低优势越大——外圈光子饥饿行是主战场。",
           "- 论文若要写“CNN 何时显著优于线性”，正确主语是 **all-32-row / 图像域**，"
           "不是 Row 16；Row 16 上只能用 §5.6 的 paired bootstrap 表述（±2% 量级、符号随能量翻转）。"]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "improvement_table.md").write_text("\n".join(md), encoding="utf-8")
    (RUNS / "_dose_improvement.json").write_text(
        json.dumps({"rows": rows, "per_energy": per_e}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("\n".join(md[:8]))
    print("...\n[done] improvement_table.md + _dose_improvement.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
