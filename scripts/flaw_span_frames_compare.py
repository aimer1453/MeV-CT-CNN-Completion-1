#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/flaw_span_frames_compare.py —— 两帧 vs 三帧输入：跨度鲁棒性是不是靠"第三帧"？

问题
----
`r34_split_attribution.md` §6 / `algo_delivered.md` R34d §4 里写过一条**推断**：

    机制是 CNN 同时看到 θ−3h / θ−h / θ+h 三帧 + sin/cos，用整条角向上下文
    约束中点，而不是只对两个端点加权，因此误差随跨度增长更缓。

验证方法当时写的是"用 `--input-views 2` 在 s=8° 重训一组，若增幅向 linear 的
62% 靠拢即成立"。本脚本就是执行并汇总这个验证。

`n_views=2` 的确切含义（`data/dataset.py` 第 330–333 行）
----------------------------------------------------------
  n_views=3（正文）：[θ−3h, θ−h, θ+h] + sin θ + cos θ → 5 通道
  n_views=2（消融）：[      θ−h, θ+h] + sin θ + cos θ → 4 通道（去掉 ch_prev）
所以两帧 = **对称包围对**，几何与三帧完全一致，只少那帧远端上下文。

数据源
------
  runs/_flaw12_span_index.csv    三帧（正文）五档？→ 实为 2/4/6/8 四档
  runs/_flaw12_span2v_index.csv  两帧（本组新训，6 run：s=2 与 s=8，后续补 4/6）
  runs/_reviewer_p0/projonly_flaw_s{step}_{run_id}.json

口径
----
与 `flaw_span_summary.py` **完全相同**：45 held-out test 视角、masked 物理域、
row16 = RECON_ROW、三个种子 20261022/23/24。因此本表可与主跨度表并列，
但**不可**与 360 待补全视角的跨能量表横比。

输出
----
  runs/_flaw12_span_frames.json
  runs/_flaw12_span_frames_table.md
  docs/figs/p01_flaw_span_frames/fig_span_frames_en.{png,pdf,csv}
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "runs" / "_reviewer_p0"
RUNS = ROOT / "runs"
FIG = ROOT / "docs" / "figs" / "p01_flaw_span_frames"
DPI = 300
STEPS = (2, 4, 6, 8)
SEEDS = (20261022, 20261023, 20261024)

#: s=2 三帧基准（现有 flaw12 3.0 MeV 三种子）
S2_RUNS_3V = {
    20261022: "20261029_2351_cnn_bs8_lr1e-4_flaw12_3.0",
    20261023: "20261030_0003_cnn_bs8_lr1e-4_flaw12_3.0",
    20261024: "20261030_0016_cnn_bs8_lr1e-4_flaw12_3.0",
}

COL = {"v3": "#B4483C", "v2": "#7A4E9E", "linear": "#3A6EA5", "cubic": "#2E8B6B"}
LBL = {"v3": "3 views: $\\theta-3h,\\ \\theta-h,\\ \\theta+h$ (main)",
       "v2": "2 views: $\\theta-h,\\ \\theta+h$ (ablation)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.4,
    "axes.linewidth": 0.7,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 5.9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.fontset": "dejavusans",
})


def index_map(name: str, seed_map: dict[int, str] | None = None
              ) -> dict[tuple[int, int], str]:
    out = dict(seed_map or {})
    p = RUNS / name
    if p.is_file():
        with p.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if (r.get("run_id") or "").strip():
                    out[(int(r["step"]), int(r["seed"]))] = r["run_id"].strip()
    return out


def load_one(step: int, rid: str) -> dict | None:
    p = SRC / f"projonly_flaw_s{step}_{rid}.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    cfg = json.loads((RUNS / rid / "config.json").read_text(encoding="utf-8"))
    proj = d["projection"]
    out = {"step": step, "run_id": rid, "seed": int(cfg["seed"]),
           "views": int(cfg.get("input_views") or 3), "epoch": d.get("cnn_epoch")}
    for m in ("cnn", "linear", "cubic"):
        out[f"row16_{m}"] = proj[m]["row16"]["rmse_valid"]
    out["row16_ratio"] = out["row16_cnn"] / out["row16_linear"]
    out["row16_ratio_cubic"] = out["row16_cnn"] / out["row16_cubic"]
    out["all32_cnn"] = proj["cnn"]["all32"]["rmse_valid"]
    out["all32_linear"] = proj["linear"]["all32"]["rmse_valid"]
    return out


def collect(idx: dict[tuple[int, int], str], views: int, missing: list[str]
            ) -> list[dict]:
    rows: list[dict] = []
    for s in STEPS:
        for sd in SEEDS:
            rid = idx.get((s, sd))
            if not rid:
                continue
            d = load_one(s, rid)
            if d is None:
                missing.append(f"{views}v s={s} seed={sd} 缺评估（{rid}）")
                continue
            if d["seed"] != sd:
                raise SystemExit(f"[FATAL] {rid} config.seed={d['seed']} != {sd}")
            if d["views"] != views:
                raise SystemExit(f"[FATAL] {rid} config.input_views={d['views']} != {views}"
                                 f"（run_id 可能被抢注）")
            rows.append(d)
    return rows


def agg_of(rows: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for s in STEPS:
        sub = [r for r in rows if r["step"] == s]
        if not sub:
            continue
        a = {"n_seeds": len(sub), "run_ids": {str(r["seed"]): r["run_id"] for r in sub}}
        for k in ("row16_cnn", "row16_linear", "row16_cubic", "row16_ratio",
                  "row16_ratio_cubic", "all32_cnn", "all32_linear"):
            v = [r[k] for r in sub]
            a[k] = {"mean": st.mean(v), "sd": st.pstdev(v) if len(v) > 1 else 0.0,
                    "values": v}
        out[s] = a
    return out


def main() -> int:
    missing: list[str] = []
    rows3 = collect(index_map("_flaw12_span_index.csv",
                              {(2, sd): rid for sd, rid in S2_RUNS_3V.items()}),
                    3, missing)
    rows2 = collect(index_map("_flaw12_span2v_index.csv"), 2, missing)
    a3, a2 = agg_of(rows3), agg_of(rows2)

    # ---------------- JSON
    payload = {
        "scope": "edge-rod defect 3_3_flaw_1_2, 3.0 MeV, projection domain, "
                 "45 held-out test views (span protocol)",
        "question": "跨度鲁棒性是否来自第三帧（θ−3h）？",
        "seeds": list(SEEDS),
        "per_seed": {"3view": rows3, "2view": rows2},
        "per_step": {"3view": {str(k): v for k, v in a3.items()},
                     "2view": {str(k): v for k, v in a2.items()}},
        "missing": missing,
    }
    (RUNS / "_flaw12_span_frames.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- Markdown
    L = ["# 两帧 vs 三帧输入：跨度鲁棒性的来源（边棒 3_3_flaw_1_2 / 3.0 MeV / 45 test 视角）", ""]
    L.append("口径与 `_flaw12_span_table.md` 完全相同（45 held-out test 视角、masked、row16、"
             "三种子 20261022/23/24）。**禁止与 360 待补全视角的跨能量表横比。**")
    L.append("")
    L.append("| s | 实测视角 | 三帧 CNN | 两帧 CNN | 三帧/两帧 | 三帧 ratio | 两帧 ratio |")
    L.append("|---|---|---|---|---|---|---|")
    for s in STEPS:
        if s not in a3 and s not in a2:
            continue
        c3 = f"{a3[s]['row16_cnn']['mean']:.6f}" if s in a3 else "—"
        c2 = f"{a2[s]['row16_cnn']['mean']:.6f}" if s in a2 else "—"
        if s in a3 and s in a2:
            rel = f"{100 * (a3[s]['row16_cnn']['mean'] / a2[s]['row16_cnn']['mean'] - 1):+.2f}%"
        else:
            rel = "—"
        r3 = (f"**{a3[s]['row16_ratio']['mean']:.4f} ± {a3[s]['row16_ratio']['sd']:.4f}**"
              if s in a3 else "—")
        r2 = (f"**{a2[s]['row16_ratio']['mean']:.4f} ± {a2[s]['row16_ratio']['sd']:.4f}**"
              if s in a2 else "—")
        L.append(f"| {s}° | {360 // s} | {c3} | {c2} | {rel} | {r3} | {r2} |")
    L.append("")
    # 增幅对照
    if 2 in a3 and 8 in a3 and 2 in a2 and 8 in a2:
        g3 = a3[8]["row16_cnn"]["mean"] / a3[2]["row16_cnn"]["mean"] - 1
        g2 = a2[8]["row16_cnn"]["mean"] / a2[2]["row16_cnn"]["mean"] - 1
        gl = a3[8]["row16_linear"]["mean"] / a3[2]["row16_linear"]["mean"] - 1
        L.append(f"2°→8° 误差增幅：三帧 **{100 * g3:+.1f}%**、两帧 **{100 * g2:+.1f}%**、"
                 f"线性插值 **{100 * gl:+.1f}%**。")
        L.append("")
    L.append("> 结论见 `r34_split_attribution.md` §7。")
    (RUNS / "_flaw12_span_frames_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    if not a2:
        print("[WARN] 两帧数据为空，未出图")
        return 0

    # ---------------- 图
    FIG.mkdir(parents=True, exist_ok=True)
    x = np.array([s for s in STEPS], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75), dpi=DPI)

    def curve(ax, agg, key, color, style, label):
        have = [s for s in STEPS if s in agg]
        mu = [agg[s][key]["mean"] for s in have]
        sd = [agg[s][key]["sd"] for s in have]
        ax.errorbar(have, mu, yerr=sd, fmt=style, color=color, lw=1.3, ms=4.0,
                    capsize=2.0, elinewidth=0.7, label=label, zorder=3)
        return have, mu

    ax = axes[0]
    ax.axhline(1.0, color="#666666", lw=0.8, zorder=1)
    h3, m3 = curve(ax, a3, "row16_ratio", COL["v3"], "-o", LBL["v3"])
    h2, m2 = curve(ax, a2, "row16_ratio", COL["v2"], "-D", LBL["v2"])
    for hh, mm, cc, dy in ((h3, m3, COL["v3"], 8), (h2, m2, COL["v2"], -12)):
        for s, v in zip(hh, mm):
            ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=6.0, color=cc)
    ax.axhspan(0.90, 1.0, color=COL["linear"], alpha=0.06, lw=0, zorder=0)
    ax.text(7.92, 0.918, "CNN better", ha="right", va="center", fontsize=6.0,
            color=COL["linear"], style="italic")
    # 两条曲线穿越 1.0 的位置（两点线性内插估计，没有 7° 的 run）
    cross = {}
    for key, agg in (("3v", a3), ("2v", a2)):
        if 6 in agg and 8 in agg:
            r6 = agg[6]["row16_ratio"]["mean"]
            r8 = agg[8]["row16_ratio"]["mean"]
            if (r6 - 1.0) * (r8 - 1.0) < 0:
                cross[key] = 6 + 2 * (r6 - 1.0) / (r6 - r8)
    if len(cross) == 2:
        ax.text(0.03, 0.965,
                f"crosses 1.0 at $\\approx${cross['2v']:.1f}$^\\circ$ (2 views)"
                f" / {cross['3v']:.1f}$^\\circ$ (3 views)",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.9,
                color="#333333")
    ax.set_xlabel("Sampling span $s$ (deg)")
    ax.set_ylabel("Row-16 RMSE ratio (CNN / linear)")
    ax.set_xticks(STEPS)
    ax.set_xticklabels([f"{s}\u00b0" for s in STEPS])
    ax.set_ylim(0.90, max(1.22, max(m2 + [1.16]) + 0.04))
    ax.set_title("(a) Does the third frame drive the crossover?", fontsize=7.6)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.95)

    ax = axes[1]
    have = [s for s in STEPS if s in a3]
    ax.errorbar(have, [a3[s]["row16_linear"]["mean"] for s in have],
                fmt="-s", color=COL["linear"], lw=1.2, ms=3.4,
                label="Linear interpolation", zorder=2)
    ax.errorbar(have, [a3[s]["row16_cubic"]["mean"] for s in have],
                fmt="-^", color=COL["cubic"], lw=1.2, ms=3.4,
                label="Cubic interpolation", zorder=2)
    curve(ax, a3, "row16_cnn", COL["v3"], "-o", LBL["v3"])
    curve(ax, a2, "row16_cnn", COL["v2"], "-D", LBL["v2"])
    ax.set_xlabel("Sampling span $s$ (deg)")
    ax.set_ylabel("Row-16 RMSE (physical units)")
    ax.set_xticks(STEPS)
    ax.set_xticklabels([f"{s}\u00b0" for s in STEPS])
    ax.set_title("(b) Absolute error of each estimator", fontsize=7.6)
    ax.legend(loc="upper left", frameon=False)

    for ax in axes:
        ax.grid(alpha=0.18, lw=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_span_frames_en.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    with open(FIG / "fig_span_frames_en.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["span_deg", "n_measured_views", "half_span_deg",
                    "cnn_row16_3view_mean", "cnn_row16_3view_sd",
                    "cnn_row16_2view_mean", "cnn_row16_2view_sd",
                    "ratio_3view_mean", "ratio_3view_sd",
                    "ratio_2view_mean", "ratio_2view_sd",
                    "linear_row16", "cubic_row16"])
        for s in STEPS:
            if s not in a3:
                continue
            d = [s, 360 // s, s // 2,
                 f"{a3[s]['row16_cnn']['mean']:.6f}", f"{a3[s]['row16_cnn']['sd']:.6f}"]
            d += ([f"{a2[s]['row16_cnn']['mean']:.6f}",
                   f"{a2[s]['row16_cnn']['sd']:.6f}"] if s in a2 else ["", ""])
            d += [f"{a3[s]['row16_ratio']['mean']:.4f}", f"{a3[s]['row16_ratio']['sd']:.4f}"]
            d += ([f"{a2[s]['row16_ratio']['mean']:.4f}",
                   f"{a2[s]['row16_ratio']['sd']:.4f}"] if s in a2 else ["", ""])
            d += [f"{a3[s]['row16_linear']['mean']:.6f}", f"{a3[s]['row16_cubic']['mean']:.6f}"]
            w.writerow(d)

    print("\n".join(L))
    print(f"\n[OK] runs/_flaw12_span_frames.json / _flaw12_span_frames_table.md / "
          f"{FIG.name}/fig_span_frames_en.png")
    if missing:
        print("[MISSING] " + "; ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
