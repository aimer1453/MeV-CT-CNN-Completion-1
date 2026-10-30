#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/flaw_span_summary.py —— 缺陷数据跨度扫描的汇总 + 折线图

回答的问题
----------
2° 下缺陷数据 CNN 0/360 全输（ratio ≈ 1.15）。如果原因是"工作点太简单
（线性插值近乎无偏）"，那么把跨度拉到 4/6/8°，线性插值会因为曲率误差崩溃，
CNN 应该反超——比值要穿过 1.0。

数据源（全部单能量 3.0 MeV、边棒 3_3_flaw_1_2、45 test 视角、masked 物理域）
--------------------------------------------------------------------------
  runs/_reviewer_p0/projonly_flaw_s{step}_{run_id}.json
    step=2：现有 `*_flaw12_3.0` 三种子（同几何，无需重训）
    step=4/6/8：`*_flawspan_s{step}_s{seed}`（见 runs/_flaw12_span_index.csv）
  每个 run 的 row16/all32 × cnn/linear/cubic 都已含在 json 里。

口径红线
--------
* 本表是 **45 个 test 视角**（span 协议）。边棒跨能量表（`_flaw12_crossenergy_table.md`）
  用的是 **360 个待补全视角**，两者**禁止横比**（数值差 ~10%）。
* 三个种子在四个跨度上完全一致（20261022/23/24），SD 是**种子间**离散度，
  不是视角间离散度。

输出
----
  runs/_flaw12_span_summary.json
  runs/_flaw12_span_table.md
  docs/figs/p01_flaw_span_sweep/fig_span_ratio_en.{png,pdf}
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
import matplotlib.patheffects as pe
import _fig_style as fs

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "runs" / "_reviewer_p0"
RUNS = ROOT / "runs"
FIG = ROOT / "docs" / "figs" / "p01_flaw_span_sweep"
DPI = 300
STEPS = (2, 4, 6, 8, 10, 12)
SEEDS = (20261022, 20261023, 20261024)

#: s=2 基准：现有 flaw12 3.0 MeV 三种子（seed -> run_id），已核 config.seed
S2_RUNS = {
    20261022: "20261029_2351_cnn_bs8_lr1e-4_flaw12_3.0",
    20261023: "20261030_0003_cnn_bs8_lr1e-4_flaw12_3.0",
    20261024: "20261030_0016_cnn_bs8_lr1e-4_flaw12_3.0",
}

# 与 Fig 8 统一用 Okabe–Ito，跨图配色一致（同一篇论文里同一方法必须同色）
COL = {"linear": "#0072B2", "cubic": "#009E73", "cnn": "#D55E00"}
LBL = {"linear": "Linear interpolation (2 views)", "cubic": "Cubic interpolation (4 views)",
       "cnn": "CNN completion"}

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


def index_map() -> dict[tuple[int, int], str]:
    """(step, seed) -> run_id：s=2 来自 S2_RUNS，其余来自跨度索引 CSV。"""
    out = {(2, sd): rid for sd, rid in S2_RUNS.items()}
    idx = RUNS / "_flaw12_span_index.csv"
    if idx.is_file():
        with idx.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if (r.get("run_id") or "").strip():
                    out[(int(r["step"]), int(r["seed"]))] = r["run_id"].strip()
    return out


def load_one(step: int, rid: str) -> dict | None:
    p = SRC / f"projonly_flaw_s{step}_{rid}.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    proj = d["projection"]
    out = {"step": step, "run_id": rid, "seed": int(json.loads(
        (RUNS / rid / "config.json").read_text(encoding="utf-8"))["seed"]),
        "epoch": d.get("cnn_epoch")}
    for roi in ("row16", "all32"):
        for m in ("cnn", "linear", "cubic"):
            out[f"{roi}_{m}"] = proj[m][roi]["rmse_valid"]
        out[f"{roi}_ratio"] = out[f"{roi}_cnn"] / out[f"{roi}_linear"]
        out[f"{roi}_ratio_cubic"] = out[f"{roi}_cnn"] / out[f"{roi}_cubic"]
    return out


def load_uniform() -> dict[int, dict]:
    """均匀域（3_3 主域）跨度扫描的 row-16 均值，供 (b) 的归一化对照。

    优先读 `runs/_reviewer_p0/sweep100_3seed_s*.json`（3 种子汇总，含 10°/12° 档，
    由 `scripts/agg_sweep_3seed.py` 产出）；缺档时回退 `runs/_r34_span_sweep.json`。
    s=2–8 两者逐位一致（0.9879/0.9890/0.9536/0.9304），可互换。
    缺档（如均匀域尚未跑 12°）直接跳过，不补造数据。
    """
    out: dict[int, dict] = {}
    for p in sorted(SRC.glob("sweep100_3seed_s*.json")):
        s = int(p.stem.split("_s")[-1])
        j = json.loads(p.read_text(encoding="utf-8"))
        proj = j["projection"]
        out[s] = {m: proj[m]["row16"]["overall"]["rmse_valid"]
                  for m in ("cnn", "linear", "cubic")}
    if out:
        return out
    p = RUNS / "_r34_span_sweep.json"
    if not p.is_file():
        return {}
    j = json.loads(p.read_text(encoding="utf-8"))
    for k, v in j.get("spans", {}).items():
        a = v["agg"]["row16"]
        out[int(k.lstrip("s"))] = {
            "cnn": float(a["cnn_mean_sd"][0]),
            "linear": float(a["linear"][0]),
            "cubic": float(a["cubic_mean_sd"][0]),
        }
    return out


def main() -> int:
    idx = index_map()
    rows: list[dict] = []
    missing: list[str] = []
    for s in STEPS:
        for sd in SEEDS:
            rid = idx.get((s, sd))
            if not rid:
                missing.append(f"s={s} seed={sd} 未登记")
                continue
            d = load_one(s, rid)
            if d is None:
                missing.append(f"s={s} seed={sd} 缺评估 json（{rid}）")
                continue
            if d["seed"] != sd:
                raise SystemExit(f"[FATAL] {rid} config.seed={d['seed']} != 期望 {sd}")
            rows.append(d)

    agg: dict[int, dict] = {}
    for s in STEPS:
        sub = [r for r in rows if r["step"] == s]
        if not sub:
            continue
        a = {"n_seeds": len(sub)}
        for k in ("row16_cnn", "row16_linear", "row16_cubic", "row16_ratio",
                  "row16_ratio_cubic", "all32_cnn", "all32_linear", "all32_cubic",
                  "all32_ratio", "all32_ratio_cubic"):
            v = [r[k] for r in sub]
            a[k] = {"mean": st.mean(v), "sd": st.pstdev(v) if len(v) > 1 else 0.0,
                    "values": v}
        a["epochs"] = [r["epoch"] for r in sub]
        a["run_ids"] = {str(r["seed"]): r["run_id"] for r in sub}
        agg[s] = a

    # ---------------- JSON
    out_json = RUNS / "_flaw12_span_summary.json"
    out_json.write_text(json.dumps({
        "scope": "edge-rod defect 3_3_flaw_1_2, 3.0 MeV, projection domain, "
                 "45 held-out test views (span protocol)",
        "mask": "(I_target>0) & (empty_proj>0), row16 = RECON_ROW",
        "seeds": list(SEEDS), "per_seed": rows,
        "per_step": {str(k): v for k, v in agg.items()},
        "missing": missing,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- Markdown
    L = ["# 缺陷数据跨度扫描（边棒 3_3_flaw_1_2，3.0 MeV，45 test 视角）", ""]
    L.append("口径：投影域 row16 = RECON_ROW，mask = `(I>0)&(E>0)`，"
             "**只在 45 个 held-out test 视角上统计**；三种子 20261022/23/24。")
    L.append("")
    L.append("| s | 实测视角 | h=s/2 | CNN row16 | linear row16 | cubic row16 "
             "| **CNN/linear** | CNN/cubic | CNN 更优? |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in STEPS:
        if s not in agg:
            L.append(f"| {s}° | {360 // s} | {s // 2}° | — | — | — | — | — | 待跑 |")
            continue
        a = agg[s]
        w = "✅ 赢" if a["row16_ratio"]["mean"] < 1 else "❌ 输"
        L.append(
            f"| {s}° | {360 // s} | {s // 2}° | "
            f"{a['row16_cnn']['mean']:.6f} ± {a['row16_cnn']['sd']:.6f} | "
            f"{a['row16_linear']['mean']:.6f} | {a['row16_cubic']['mean']:.6f} | "
            f"**{a['row16_ratio']['mean']:.4f} ± {a['row16_ratio']['sd']:.4f}** | "
            f"{a['row16_ratio_cubic']['mean']:.4f} | {w} |")
    L.append("")
    L.append("all32（全 32 行）对照：")
    L.append("")
    L.append("| s | CNN all32 | linear all32 | cubic all32 | **CNN/linear** | **CNN/cubic** |")
    L.append("|---|---|---|---|---|---|")
    for s in STEPS:
        if s in agg:
            a = agg[s]
            L.append(f"| {s}° | {a['all32_cnn']['mean']:.6f} ± {a['all32_cnn']['sd']:.6f} | "
                     f"{a['all32_linear']['mean']:.6f} | "
                     f"{a['all32_cubic']['mean']:.6f} | "
                     f"**{a['all32_ratio']['mean']:.4f} ± {a['all32_ratio']['sd']:.4f}** | "
                     f"**{a['all32_ratio_cubic']['mean']:.4f} ± {a['all32_ratio_cubic']['sd']:.4f}** |")
    L.append("")
    if missing:
        L.append("**缺项**：" + "; ".join(missing))
        L.append("")
    L.append("> 红线：本表是 **45 test 视角**；边棒跨能量表是 **360 待补全视角**，"
             "两者禁止横比。")
    (RUNS / "_flaw12_span_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---------------- 图
    FIG.mkdir(parents=True, exist_ok=True)
    have = [s for s in STEPS if s in agg]
    if not have:
        print("[WARN] 无数据，未出图")
        return 0
    x = np.array(have, dtype=float)
    uni = load_uniform()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.85), dpi=DPI)

    # (a) 两域 CNN/linear 比值：缺陷域（实线朱红）+ 均匀域（虚线蓝）
    ax = axes[0]
    mu = [agg[s]["row16_ratio"]["mean"] for s in have]
    sd = [agg[s]["row16_ratio"]["sd"] for s in have]
    ax.axhline(1.0, color="#666666", lw=0.8, ls="-", zorder=1)
    # 缺陷域：SD 带 + 主曲线 + 逐种子点
    ax.fill_between(x, np.subtract(mu, sd), np.add(mu, sd), color=COL["cnn"],
                    alpha=0.16, lw=0, zorder=2)
    ax.plot(x, mu, "-o", color=COL["cnn"], lw=1.3, ms=4.2, zorder=4,
            label="Defect (edge rod)")
    for s in have:
        ax.plot([s] * len(agg[s]["row16_ratio"]["values"]),
                agg[s]["row16_ratio"]["values"], ls="none", marker="o", ms=2.0,
                mfc="white", mec=COL["cnn"], mew=0.7, zorder=5)
    # 均匀域比值（虚线蓝）
    hu = [s for s in have if s in uni]
    if hu:
        ru = [uni[s]["cnn"] / uni[s]["linear"] for s in hu]
        ax.plot(hu, ru, "--s", color=COL["linear"], lw=1.1, ms=3.4, zorder=4,
                mfc="white", mec=COL["linear"], mew=0.8, label="Uniform array")
    # 缺陷域数值标签（数据派生）
    for s in have:
        v = agg[s]["row16_ratio"]["mean"]
        above = v > 1.0
        ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                    xytext=(0, 8 if above else -11), ha="center",
                    va="bottom" if above else "top",
                    fontsize=6.0, color=COL["cnn"],
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    # y 轴范围 + 下方"CNN 更优"淡蓝带
    lo = min(min(np.subtract(mu, sd)), min(v for s in have
                                          for v in agg[s]["row16_ratio"]["values"]))
    hi = max(max(np.add(mu, sd)), max(v for s in have
                                      for v in agg[s]["row16_ratio"]["values"]))
    y_lo = min(lo - 0.03, 0.76)
    y_hi = max(min(hi + 0.055, 1.22), 1.02)
    ax.set_ylim(y_lo, y_hi)
    ax.axhspan(y_lo, 1.0, color=COL["linear"], alpha=0.06, lw=0, zorder=0)
    # 缺陷域穿越 1.0 的跨度（6°→8° 线性内插），用 ★ 标出；
    # 精确值由 CSV/图注承载，图上不写 "6.9°" 之类误导精度
    if 6 in agg and 8 in agg:
        r6, r8 = agg[6]["row16_ratio"]["mean"], agg[8]["row16_ratio"]["mean"]
        if (r6 - 1.0) * (r8 - 1.0) < 0:
            s_star = 6 + 2 * (r6 - 1.0) / (r6 - r8)
            ax.axvline(s_star, color="#666666", lw=0.7, ls=(0, (2, 2)), zorder=1)
            ax.annotate("$\\bigstar$", (s_star, 1.0), fontsize=12.0,
                        color="#1A1A1A", ha="center", va="center", zorder=7)
    # 局部放大：均匀域小跨度（2°→4°）CNN/linear 比值 0.988→0.989，统计上 unresolved
    # 放在上栏右上方空白区（s>7、ratio>1.05 处无主数据），避免压到曲线
    axin = ax.inset_axes([0.52, 0.50, 0.44, 0.42])
    axin.patch.set_facecolor("white")
    axin.patch.set_alpha(0.95)
    su = [s for s in STEPS if s in uni and s <= 6]
    if su:
        rv = [uni[s]["cnn"] / uni[s]["linear"] for s in su]
        axin.axhline(1.0, color="#888888", lw=0.6)
        axin.plot(su, rv, "-o", color=COL["linear"], lw=1.1, ms=3.2)
        for s, v in zip(su, rv):
            axin.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                          xytext=(0, 7), fontsize=5.2, ha="center",
                          color=COL["linear"])
        axin.set_xlim(1.4, 6.3)
        axin.set_ylim(0.977, 0.997)
        axin.set_xticks([2, 4, 6])
        axin.set_xticklabels(["2°", "4°", "6°"], fontsize=5.0)
        axin.tick_params(labelsize=5.0)
        axin.set_title("uniform, small span", fontsize=5.4)
        axin.grid(alpha=0.2, lw=0.4)
        axin.annotate("unresolved (Δ≈0.001)", (4, 0.989),
                      textcoords="offset points", xytext=(2, -13),
                      fontsize=5.0, color="#333333", ha="left")
        ax.indicate_inset_zoom(axin, edgecolor="#666666", alpha=0.55, lw=0.7)
    ax.set_xlabel("Sampling span $s$ (deg)")
    ax.set_ylabel("Row-16 RMSE ratio (CNN / linear)")
    ax.set_xticks(STEPS)
    ax.set_xticklabels([f"{s}°" for s in STEPS])
    ax.set_title("(a) Completion gain vs. sampling span", fontsize=7.6)
    ax.legend(loc="upper center", ncol=2, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.95, fontsize=6.0)

    # (b) 归一化退化率：每方法 RMSE(s) / 自身 RMSE(s=2)，缺陷域与均匀域同图
    ax = axes[1]
    domains = []
    if all(f"row16_{m}" in agg[have[0]] for m in ("linear", "cubic", "cnn")):
        domains.append(("Defect (edge rod)", "-", "o", have,
                        {m: [agg[s][f"row16_{m}"]["mean"] for s in have]
                         for m in ("linear", "cubic", "cnn")}))
    hu = [s for s in have if s in uni]
    if len(hu) >= 2:
        domains.append(("Uniform array", (0, (4, 2)), "s", hu,
                        {m: [uni[s][m] for s in hu] for m in ("linear", "cubic", "cnn")}))
    for name, ls, mk, ss, ser in domains:
        for m in ("linear", "cubic", "cnn"):
            yv = np.asarray(ser[m], dtype=float)
            yv = yv / yv[0]                      # 归一到各自 s=2，无量纲
            ax.plot(ss, yv, ls=ls, marker=mk, color=COL[m],
                    lw=(1.35 if m == "cnn" else 1.05), ms=(3.4 if m == "cnn" else 3.0),
                    zorder=(4 if m == "cnn" else 3))
    # 缺陷域两端的增幅直接标在线上（跨 2°→最宽档的全域增幅）
    if domains:
        ss, ser = domains[0][3], domains[0][4]
        for m, dy in (("linear", 4), ("cnn", -11)):
            yv = np.asarray(ser[m], dtype=float)
            ax.annotate(f"{(yv[-1] / yv[0] - 1) * 100:+.0f}%", (ss[-1], yv[-1] / yv[0]),
                        textcoords="offset points", xytext=(-3, dy), fontsize=5.9,
                        color=COL[m], ha="right", va="bottom" if dy > 0 else "top",
                        path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    ax.margins(y=0.10)                       # 给端点标签留出上下余量
    # 方法图例：仅 Linear / Cubic / CNN 三项（外加域虚线由下方第二条图例说明）
    handles_m = [plt.Line2D([], [], color=COL[m], lw=(1.35 if m == "cnn" else 1.05),
                            marker="o", ms=3.2, label=LBL[m])
                 for m in ("linear", "cubic", "cnn")]
    h_m = ax.legend(handles=handles_m, loc="upper left", frameon=False)
    ax.add_artist(h_m)
    if len(domains) == 2:
        ax.legend(handles=[plt.Line2D([], [], color="#555555", ls="-", lw=1.1,
                                      label="Defect domain"),
                           plt.Line2D([], [], color="#555555", ls=(0, (4, 2)), lw=1.1,
                                      label="Uniform domain")],
                  loc="lower right", frameon=False)
    ax.set_xlabel("Sampling span $s$ (deg)")
    ax.set_ylabel("RMSE / RMSE($s$=2°)  (dimensionless)")
    ax.set_xticks(STEPS)
    ax.set_xticklabels([f"{s}°" for s in STEPS])
    ax.set_title("(b) Normalised degradation of each estimator", fontsize=7.6)

    for ax in axes:
        ax.grid(alpha=0.18, lw=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    # 仅输出 PNG（审稿要求 PNG only，禁止 PDF）
    fig.savefig(FIG / "fig_span_ratio_en.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # 绘图数据落盘（论文侧核对用；每个数字都能追到 per_step / per_seed）
    # 新增 norm_* 三列 = 各方法自己的 RMSE(s)/RMSE(2°)，即 (b) 的画图数据；
    # 新增 uniform_* 三列 = 均匀域同协议对照（N/A 表示该档尚未训练）。
    uni = load_uniform()
    s0 = have[0]
    base = {m: agg[s0][f"row16_{m}"]["mean"] for m in ("cnn", "linear", "cubic")}
    u0 = uni.get(min(uni)) if uni else None
    with open(FIG / "fig_span_ratio_en.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["span_deg", "n_measured_views", "half_span_deg",
                    "row16_cnn_mean", "row16_cnn_sd", "row16_linear", "row16_cubic",
                    "ratio_cnn_linear_mean", "ratio_cnn_linear_sd",
                    "ratio_cnn_cubic_mean", "n_seeds",
                    "norm_cnn", "norm_linear", "norm_cubic",
                    "uniform_row16_cnn", "uniform_row16_linear", "uniform_row16_cubic",
                    "uniform_ratio_cnn_linear", "uniform_norm_cnn",
                    "uniform_norm_linear", "uniform_norm_cubic"])
        for s in have:
            a = agg[s]
            u = uni.get(s)
            w.writerow([
                s, 360 // s, s // 2,
                f"{a['row16_cnn']['mean']:.6f}", f"{a['row16_cnn']['sd']:.6f}",
                f"{a['row16_linear']['mean']:.6f}", f"{a['row16_cubic']['mean']:.6f}",
                f"{a['row16_ratio']['mean']:.4f}", f"{a['row16_ratio']['sd']:.4f}",
                f"{a['row16_ratio_cubic']['mean']:.4f}", a["n_seeds"],
                f"{a['row16_cnn']['mean'] / base['cnn']:.4f}",
                f"{a['row16_linear']['mean'] / base['linear']:.4f}",
                f"{a['row16_cubic']['mean'] / base['cubic']:.4f}",
                f"{u['cnn']:.6f}" if u else "N/A",
                f"{u['linear']:.6f}" if u else "N/A",
                f"{u['cubic']:.6f}" if u else "N/A",
                f"{u['cnn'] / u['linear']:.4f}" if u else "N/A",
                f"{u['cnn'] / u0['cnn']:.4f}" if u else "N/A",
                f"{u['linear'] / u0['linear']:.4f}" if u else "N/A",
                f"{u['cubic'] / u0['cubic']:.4f}" if u else "N/A",
            ])

    print("\n".join(L))
    print(f"\n[OK] {out_json.name} / _flaw12_span_table.md / "
          f"{FIG.name}/fig_span_ratio_en.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
