#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/r34_dose_scope_check.py —— ③ Fig 9 f=1.0 与 Table 5 差异的定量归因

问题（论文侧提出）
------------------
Fig 9 的 f=1.0 点 row16 CNN = 0.1101，而 Table 5 的 3.0 MeV 行 CNN = 0.0912
（最后一行 cnn1d 七种子是 0.0875）。看起来"同一能量同剂量却差 20%"。

本脚本把这一差异拆成三个可加/可乘的因子，并逐项定量：

  (1) 模型不同   cnn1d（1D 单行 U-Net，C 组，七种子） vs cnn（2D 全模型，D 组）
  (2) 视角集不同 45 个 test 视角（Table 5） vs 全部 180 个待补全视角（Fig 9）
  (3) 池化不同   单档 3.0 MeV vs 四档简单平均（1.5 MeV 通道把均值拉高）

根因定位（代码层）
------------------
`scripts/dose_eval.py` 在聚合 `agg`（→ recon_domain 的 row16/all32 均值）时，
`slot["row16"].append(...)` 写在 `if split_of.get(int(a)) != "test": continue`
**之前**，因此 Fig 9 的 row16/all32 是"全部待补全视角"，只有 `per_sample_test`
（另存）才是 45 个 test 视角。

用法
----
    python scripts/r34_dose_scope_check.py
输出
----
    runs/_r34_dose_scope.json  + 控制台表格
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DOSE_JSON = RUNS / "_dose_eval_f1.json"
FIG_SUMMARY = ROOT / "docs" / "figs" / "p01_dose_rmse" / "fig_dose_rmse_summary.json"
STEP1_JSON = RUNS / "_step1_row16_seeds.json"
TABLE5_JSON = RUNS / "_projdomain_metrics.json"   # 若存在则用于交叉核对

CNN_RUNS = ["20261024_0400_cnn_bs8_lr1e-4",
            "20261024_1633_cnn_bs8_lr1e-4",
            "20261024_1805_cnn_bs8_lr1e-4"]
HEADLINE = CNN_RUNS[0]
ENERGIES = ["1.5", "3", "4.5", "6"]


def quad(vals) -> float:
    v = list(vals)
    return math.sqrt(sum(x * x for x in v) / len(v))


def main() -> int:
    d = json.loads(DOSE_JSON.read_text(encoding="utf-8"))
    fig = json.loads(FIG_SUMMARY.read_text(encoding="utf-8"))

    # ---- (2) 视角集效应：同一 seed、同能量、同模型，45-view vs 180-view
    e3 = d["energies"]["3"]
    ps = e3["per_sample_test"]
    rd = e3["recon_domain"]

    def test_only(run):
        return quad([p[run] for p in ps])

    view_eff = {}
    for r in CNN_RUNS + ["linear_interp", "cubic_interp"]:
        if r == "linear_interp":
            t = quad([p["linear_interp"] for p in ps])
        elif r == "cubic_interp":
            t = quad([p["cubic_interp"] for p in ps])
        else:
            t = test_only(r)
        a = rd[r]["row16"]
        view_eff[r] = {"test45": t, "all180": a, "ratio": a / t}
    cnn_test45_mean = sum(test_only(r) for r in CNN_RUNS) / len(CNN_RUNS)
    cnn_all180_mean = sum(rd[r]["row16"] for r in CNN_RUNS) / len(CNN_RUNS)

    # ---- (3) 池化效应：四档简单平均 vs 3.0 MeV 单档
    per_e_all180 = {}
    for e in ENERGIES:
        rde = d["energies"][e]["recon_domain"]
        per_e_all180[e] = {
            "cnn": sum(rde[r]["row16"] for r in CNN_RUNS) / len(CNN_RUNS),
            "linear": rde["linear_interp"]["row16"],
            "cubic": rde["cubic_interp"]["row16"]}
    pool_cnn = sum(per_e_all180[e]["cnn"] for e in ENERGIES) / len(ENERGIES)
    pool_lin = sum(per_e_all180[e]["linear"] for e in ENERGIES) / len(ENERGIES)

    fig_pool_cnn = float(fig["row16"]["cnn_point"][0])
    fig_pool_lin = float(fig["row16"]["linear"][0])

    # ---- (1) 模型效应：cnn1d 七种子(3MeV,45test) vs 2D（同口径）
    cnn1d = None
    if STEP1_JSON.is_file():
        s1 = json.loads(STEP1_JSON.read_text(encoding="utf-8"))
        vals = [v for v in s1["seeds"].values() if v is not None]
        cnn1d = {"mean": sum(vals) / len(vals),
                 "linear": s1["linear_row16_rmse"], "n_seeds": len(vals)}

    # ---- 乘法分解链：0.0875(1D,45v) → 0.1101(2D,180v,pooled)
    chain = []
    if cnn1d:
        chain.append(("A. cnn1d 七种子 3.0MeV 45-test", cnn1d["mean"]))
        chain.append(("B. 2D 全模型 3.0MeV 45-test (3seed)", cnn_test45_mean))
        chain.append(("C. 2D 全模型 3.0MeV 180-view (3seed)", cnn_all180_mean))
        chain.append(("D. 2D 全模型 四档简单平均 180-view", pool_cnn))

    print("=" * 78)
    print("③ Fig 9 f=1.0 与 Table 5 差异归因（3.0 MeV 为主线）")
    print("=" * 78)
    print("\n[根因] dose_eval.py 的 agg 聚合发生在 split=='test' 过滤之前，")
    print("       故 recon_domain(row16/all32) 是【全部 180 个待补全视角】，")
    print("       不是 45 个 test 视角（后者另存为 per_sample_test）。\n")

    print("[因子 2] 视角集效应（同 seed / 同能量 / 同模型）")
    print("  %-34s %10s %10s %8s" % ("method", "45-test", "180-all", "ratio"))
    for k, v in view_eff.items():
        print("  %-34s %10.6f %10.6f %8.4f" % (k, v["test45"], v["all180"], v["ratio"]))
    print("  → 视角集只解释约 %+.1f%%，不是主因。"
          % (100 * (cnn_all180_mean / cnn_test45_mean - 1)))

    print("\n[因子 3] 池化效应（四档简单平均）")
    print("  %-6s %10s %10s %10s" % ("E/MeV", "CNN", "linear", "cubic"))
    for e in ENERGIES:
        p = per_e_all180[e]
        print("  %-6s %10.6f %10.6f %10.6f" % (e, p["cnn"], p["linear"], p["cubic"]))
    print("  %-6s %10.6f %10.6f" % ("mean", pool_cnn, pool_lin))
    print("  Fig9 summary f=1.0 CNN=%.6f linear=%.6f  (差 %.1e / %.1e)"
          % (fig_pool_cnn, fig_pool_lin, abs(pool_cnn - fig_pool_cnn), abs(pool_lin - fig_pool_lin)))
    print("  → 四档平均把 3.0 MeV 的 %.6f 抬到 %.6f（%+.1f%%），因为 1.5 MeV 通道 0.19"
          % (per_e_all180["3"]["cnn"], pool_cnn, 100 * (pool_cnn / per_e_all180["3"]["cnn"] - 1)))

    if cnn1d:
        print("\n[因子 1] 模型效应（同为 3.0 MeV / 45-test）")
        print("  cnn1d(1D 单行, %d seeds) = %.6f" % (cnn1d["n_seeds"], cnn1d["mean"]))
        print("  cnn  (2D 全模型, 3 seeds) = %.6f   → %+.1f%%"
              % (cnn_test45_mean, 100 * (cnn_test45_mean / cnn1d["mean"] - 1)))

        print("\n[乘法分解链]")
        prev = None
        for name, val in chain:
            step = "" if prev is None else "  ×%.4f (%+.1f%%)" % (val / prev, 100 * (val / prev - 1))
            print("  %-40s %.6f%s" % (name, val, step))
            prev = val
        print("  %-40s %+.1f%%" % ("A→D 总变化", 100 * (pool_cnn / cnn1d["mean"] - 1)))

    print("\n[Table 5 交叉核对]")
    print("  Table5 3.0MeV 行: linear 0.0884 / CNN 0.0912 → 对应 %s 的 45-test"
          % HEADLINE)
    print("    实算: linear %.6f / CNN %.6f" % (quad([p["linear_interp"] for p in ps]),
                                              test_only(HEADLINE)))
    print("  Table5 末行 'Row16, 3.0MeV (seven seeds)': 0.0884 / 0.0875 → 这是 cnn1d，不是 2D")

    print("\n[结论] Fig 9 的 f=1.0 点不能与 Table 5 任一单能量行直接比较：")
    print("       它同时换了【模型】【视角集】【能量池化】三者。")
    print("       若要与 Table 5 对齐，Fig 9 图注须写明：四档简单平均 + 全部 180 个待补全视角。")

    out = {
        "note": "Fig9 f=1.0 vs Table5 差异归因；recon_domain 为全部 180 待补全视角",
        "root_cause": "dose_eval.py 的 agg.append 位于 split=='test' 过滤之前",
        "view_set_effect": view_eff,
        "per_energy_all180": per_e_all180,
        "pool_simple_mean": {"cnn": pool_cnn, "linear": pool_lin},
        "fig9_summary_f1": {"cnn": fig_pool_cnn, "linear": fig_pool_lin},
        "cnn1d_seven_seed": cnn1d,
        "chain": [{"name": n, "value": v} for n, v in chain],
    }
    p = RUNS / "_r34_dose_scope.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
