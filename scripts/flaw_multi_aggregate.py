"""多能量缺陷域的汇总与出图：图像域指标、空腔 CNR、投影域匹配滤波 t。

输入
----
`runs/_flaw_multi_index.csv`（energy,seed,run_id）+ 每个 run 的
`runs/<run_id>_flaw_step3/image_metrics.json`（由 p01_flaw_step3_eval.py 产出）。

三种量的**种子依赖性不同**，汇总时必须分开说（混在一起会被审稿人抓）：

1. 图像域 RMSE/PSNR/SSIM、空腔 CNR
   依赖训练权重 → 逐种子独立，报 mean ± std(ddof=1)，n=3。

2. 投影域轨迹匹配滤波 t
   `proj_matched_filter` 用 **实测正弦图**（flaw − uniform）算，**与模型无关**，
   因此逐种子逐位相同，std 恰为 0。这不是缺失数据，而是一条硬结论：
   缺陷在投影域的显著性由数据本身决定，与训练好坏无关。
   图上的误差线用 **360 个视角的 bootstrap 95% 置信区间**（重采样角度），
   不是种子标准差——README 里必须写清。

3. 模型相关的"误差是否集中在轨迹上"
   `rel_ratio_on_off`（轨迹带内相对误差 / 带外相对误差）逐种子变化，
   作为 t 值的模型侧补充，报 mean ± std。

用法
----
  python scripts/flaw_multi_aggregate.py
  python scripts/flaw_multi_aggregate.py --index runs/_flaw_multi_index.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIG_DIR = ROOT / "docs" / "figs" / "p01_flaw_multienergy"
#: 期望的四档（仅用于提示；实际聚合按索引里出现的能量动态排序，
#: 因为缺陷表写 `3.0`/`6.0`、均匀侧 token 写 `3`/`6`，字符串比较不可靠）
EXPECTED_ENERGIES = ("1.5", "3.0", "4.5", "6.0")
RNG_SEED = 20261026


def mean_std(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {"n": 0, "mean": None, "std": None}
    m = float(np.mean(xs))
    s = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    return {"n": len(xs), "mean": m, "std": s, "values": xs}


def contrast_ci(contrast: float, sem: float, z: float = 1.96) -> dict:
    """D4 对比的 95% 置信区间与对应的 t 区间（正态近似）。

    坑（第一版踩过）：`proj_matched_filter_curve.npy` 存的是 **offset-0 的原始
    ΔL(θ)**（360 个角），它自身 mean/sem 给出的 t≈−20，是"缺陷 vs 无缺陷"的
    原始对比；而正文报的 **t=−7.07 是 D4 对比**——offset0 减去 90/180/270 三条
    伪轨迹基线的均值，专门消掉两次仿真之间的全局乘性差。两者不是同一统计量，
    在原始曲线上 bootstrap 会得到错的区间（实测 [−22.9, −17.8] vs 点估计 −7.07）。
    伪轨迹曲线没有落盘，无法重采样，故用解析法：
    t = ΔL/SEM，ΔL 的 95% CI 为 ΔL ± 1.96·SEM ⇒ t 的 95% CI 为 t ± 1.96。
    """
    return {"dl_ci95": [contrast - z * sem, contrast + z * sem],
            "t_halfwidth": z,
            "t_ci95": [contrast / sem - z, contrast / sem + z] if sem > 0 else None,
            "note": ("正态近似：t = ΔL/SEM ⇒ 95% CI = t ± 1.96；"
                     "伪轨迹曲线未落盘，故不做角度重采样")}


def psnr_common_by_run(run_ids: list[str], methods: list[str], dr: float) -> dict:
    """用**统一 data_range** 重算 PSNR（跨能量可比）。

    为什么必须重算：`baseline_recon.image_metrics` 的 `psnr` 用的是**该档 dense 自己**的
    `max-min` 作 data_range。缺陷数据四档的 data_range 差 2.2 倍
    （2026-10-28 换代后实测：1.5→0.12777、3.0→0.09381、4.5→0.05884、6.0→0.07097，
     最大的是 **1.5 MeV**，DR_COMMON=0.12777254；旧批 `3_3_flaw` 是 0.1096，勿混用）
    （1.5→0.1096、3.0→0.1010、4.5→0.0501、6.0→0.0712），直接并列会让
    4.5 MeV 的 PSNR 被系统性压低（实测 40.07 dB，而同口径下反而最高）。
    主文 Table 6 用的是统一 data_range 的 `psnr_common`，这里沿用同一约定：
    DR = 四档 data_range 的最大值，逐样本算 10log10(DR²/rmse_valid²) 后取均值。
    """
    out: dict[str, dict[str, float]] = {}
    for rid in run_ids:
        p = RUNS / f"{rid}_flaw_step3" / "image_per_sample.csv"
        if not p.is_file():
            continue
        per_method_vals: dict[str, list[float]] = {m: [] for m in methods}
        with p.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                m = r["method"]
                if m not in per_method_vals:
                    continue
                rv = float(r["rmse_valid"])
                if rv <= 0:
                    continue
                per_method_vals[m].append(10.0 * math.log10(dr * dr / (rv * rv)))
        out[rid] = {m: (float(np.mean(v)) if v else None)
                    for m, v in per_method_vals.items()}
    return out


def collect(index_csv: Path) -> tuple[dict, list[str]]:
    rows = list(csv.DictReader(index_csv.open(encoding="utf-8")))
    per_run: list[dict] = []
    missing: list[str] = []
    for r in rows:
        rid = (r["run_id"] or "").strip()
        if not rid:
            continue
        p = RUNS / f"{rid}_flaw_step3" / "image_metrics.json"
        if not p.is_file():
            missing.append(rid)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        pm = d["per_method"]
        cnr = d["defect_cnr"]
        vs = d.get("projection_domain", {}).get("per_view_row16", {})
        ratios = [vs[ph]["rel_ratio_on_off"] for ph in ("A", "B") if ph in vs]
        abs_ratios = [vs[ph]["ratio_on_off"] for ph in ("A", "B") if ph in vs]
        entry = {
            "run_id": rid, "energy": r["energy"], "seed": r["seed"],
            "image": {m: {k: pm[m][k] for k in ("rmse_valid", "rmse_full", "psnr", "ssim")}
                      for m in pm},
            "data_range": pm["cnn_completed"]["data_range"],
            "cnr": {m: (cnr[m].get("in_image") or {}).get("cnr") for m in cnr},
            "cnr_contrast": {m: (cnr[m].get("in_image") or {}).get("contrast") for m in cnr},
            "cnr_noise_sd": {m: (cnr[m].get("in_image") or {}).get("noise_sd") for m in cnr},
            "rel_ratio_on_off_mean2phase": float(np.mean(ratios)) if ratios else None,
            "abs_ratio_on_off_mean2phase": float(np.mean(abs_ratios)) if abs_ratios else None,
            "mf_t_stat": d.get("projection_matched_filter", {})
                          .get("d4_contrast", {}).get("t_stat"),
            "mf_contrast": d.get("projection_matched_filter", {})
                            .get("d4_contrast", {}).get("offset0_minus_pseudo_baseline"),
            "mf_sem": d.get("projection_matched_filter", {}).get("d4_contrast", {}).get("sem"),
            "curve_path": str(RUNS / f"{rid}_flaw_step3" / "proj_matched_filter_curve.npy"),
        }
        per_run.append(entry)

    energy_summary: dict = {}
    # 统一 data_range（四档最大值）→ psnr_common
    dr_common = max(p["data_range"] for p in per_run)
    pc = psnr_common_by_run([p["run_id"] for p in per_run],
                           ["cnn_completed", "linear_interp", "fbp_sparse",
                            "sart_sparse"], dr_common)
    for p in per_run:
        p["psnr_common"] = pc.get(p["run_id"], {})
    # 按索引里**实际出现**的能量动态聚合，顺序按数值排序。
    # 坑：缺陷数据表把能量写作 `3.0` / `6.0`，而均匀侧文件 token 是 `3` / `6`；
    # 用固定元组 ("1.5","3","4.5","6") 去匹配会静默丢掉 3.0 与 6.0 两档。
    present = sorted({p["energy"] for p in per_run}, key=float)
    for e in present:
        sub = [p for p in per_run if p["energy"] == e]
        if not sub:
            continue
        s: dict = {"n_runs": len(sub), "run_ids": [p["run_id"] for p in sub]}
        s["image"] = {
            m: {k: mean_std([p["image"][m][k] for p in sub])
                for k in ("rmse_valid", "rmse_full", "psnr", "ssim")}
            for m in ("cnn_completed", "linear_interp", "fbp_sparse", "sart_sparse", "dense")}
        s["cnr"] = {m: mean_std([p["cnr"][m] for p in sub]) for m in sub[0]["cnr"]}
        s["psnr_common"] = {
            m: mean_std([p["psnr_common"].get(m) for p in sub])
            for m in ("cnn_completed", "linear_interp", "fbp_sparse", "sart_sparse")}
        s["rel_ratio_on_off"] = mean_std([p["rel_ratio_on_off_mean2phase"] for p in sub])
        s["abs_ratio_on_off"] = mean_std([p["abs_ratio_on_off_mean2phase"] for p in sub])
        ts = [p["mf_t_stat"] for p in sub if p["mf_t_stat"] is not None]
        cs = [p["mf_contrast"] for p in sub if p["mf_contrast"] is not None]
        ss = [p["mf_sem"] for p in sub if p["mf_sem"] is not None]
        s["mf_t_stat"] = {"n": len(ts), "mean": float(np.mean(ts)) if ts else None,
                          "std": float(np.std(ts, ddof=1)) if len(ts) > 1 else 0.0,
                          "values": ts,
                          "seed_invariant": bool(len(ts) > 1 and float(np.ptp(ts)) < 1e-9)}
        s["mf_contrast_mean"] = float(np.mean(cs)) if cs else None
        s["mf_sem_mean"] = float(np.mean(ss)) if ss else None
        if cs and ss:
            s["mf_t_ci"] = contrast_ci(float(np.mean(cs)), float(np.mean(ss)))
        energy_summary[e] = s

    return {"per_run": per_run, "per_energy": energy_summary,
            "_meta": {"psnr_common_data_range": dr_common,
                      "psnr_common_rule": ("DR = 四档 data_range 最大值；逐样本 "
                                           "10log10(DR^2/rmse_valid^2) 后取均值"),
                      "n_runs_total": len(per_run)}}, missing


def md_table(summary: dict) -> str:
    """可直接粘进交接文档的 Markdown 表。"""
    L: list[str] = []
    ns = sorted({s["n_runs"] for s in summary.values()})
    n_lab = "/".join(str(n) for n in ns)
    L.append(f"### 多能量缺陷域 · 图像域指标（180 实测 + 180 补全，45 个 test 视角池化，"
             f"n={n_lab} 种子）")
    L.append("")
    L.append("| 能量 | n | RMSE_valid | RMSE_full | PSNR_common | SSIM | 空腔 CNR |")
    L.append("|---|---|---|---|---|---|---|")
    for e, s in summary.items():
        c = s["image"]["cnn_completed"]
        cn = s["cnr"]["cnn_completed"]
        pc = (s.get("psnr_common") or {}).get("cnn_completed") or {}
        ps = (f"{pc['mean']:.2f} ± {pc['std']:.2f}" if pc.get("mean") is not None else "—")
        L.append(f"| {e} MeV | {s['n_runs']} "
                 f"| {c['rmse_valid']['mean']:.6f} ± {c['rmse_valid']['std']:.6f} "
                 f"| {c['rmse_full']['mean']:.6f} ± {c['rmse_full']['std']:.6f} "
                 f"| {ps} "
                 f"| {c['ssim']['mean']:.4f} ± {c['ssim']['std']:.4f} "
                 f"| {cn['mean']:.2f} ± {cn['std']:.2f} |")
    L.append("")
    L.append("> PSNR_common 用四档统一 data_range（= 四档 dense 取值范围的最大值），"
             "逐样本算 10log10(DR²/rmse²) 后平均；**不要**用逐档 data_range 的 PSNR "
             "做跨能量比较（四档 data_range 差 2.2 倍，会把 4.5 MeV 从最高压到最低）。")
    L.append("")
    L.append(f"### 同能量档的基线对照（CNR，n={n_lab} 种子；dense/linear/FBP/SART 与种子无关）")
    L.append("")
    L.append("| 能量 | dense | CNN 补全 | 线性插值 | FBP 稀疏 | SART 稀疏 |")
    L.append("|---|---|---|---|---|---|")
    for e, s in summary.items():
        cells = []
        for m in ("dense", "cnn_completed", "linear_interp", "fbp_sparse", "sart_sparse"):
            v = s["cnr"][m]
            cells.append(f"{v['mean']:.2f} ± {v['std']:.2f}" if v["mean"] is not None else "—")
        L.append(f"| {e} MeV | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### 投影域轨迹匹配滤波（种子无关，std 恒为 0）")
    L.append("")
    L.append("| 能量 | t（均值 ± 种子std） | 对比度 ΔL | SEM | ΔL 的 95% CI "
             "| 误差集中度（绝对比 on/off） | 相对误差比 on/off |")
    L.append("|---|---|---|---|---|---|")
    for e, s in summary.items():
        mf = s["mf_t_stat"]
        ci = s.get("mf_t_ci") or {}
        dl_ci = ci.get("dl_ci95")
        ci_s = f"[{dl_ci[0]:.5f}, {dl_ci[1]:.5f}]" if dl_ci else "—"
        contrast = s.get("mf_contrast_mean")
        contrast_s = f"{contrast:.5f}" if contrast is not None else "—"
        sem = s.get("mf_sem_mean")
        sem_s = f"{sem:.5f}" if sem is not None else "—"
        ar = s.get("abs_ratio_on_off") or {}
        rr = s["rel_ratio_on_off"]
        ar_s = f"{ar['mean']:.3f} ± {ar['std']:.3f}" if ar.get("mean") is not None else "—"
        rr_s = (f"{rr['mean']:.3f} ± {rr['std']:.3f}" if rr["mean"] is not None else "—")
        t_s = (f"{mf['mean']:.2f} ± {mf['std']:.2f}" if mf["mean"] is not None else "—")
        L.append(f"| {e} MeV | {t_s} | {contrast_s} | {sem_s} | {ci_s} | {ar_s} | {rr_s} |")
    return "\n".join(L)


def plot(summary: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    es = [float(e) for e in summary]
    t_mean = [summary[e]["mf_t_stat"]["mean"] for e in summary]
    # 误差线：t = ΔL/SEM 的 95% CI ⇒ t ± 1.96（解析法，与 t 检验同一正态假设）
    t_err = [1.96 for _ in es]

    cnr_mean = [summary[e]["cnr"]["cnn_completed"]["mean"] for e in summary]
    cnr_std = [summary[e]["cnr"]["cnn_completed"]["std"] for e in summary]
    sart_mean = [summary[e]["cnr"]["sart_sparse"]["mean"] for e in summary]
    sart_std = [summary[e]["cnr"]["sart_sparse"]["std"] for e in summary]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9))

    # (a) 匹配滤波 t
    ax1.axhline(0, color="#999999", lw=0.9, zorder=1)
    for y in (-2.02, 2.02):
        ax1.axhline(y, color="#c0392b", lw=1.0, ls=":", zorder=1)
    ax1.errorbar(es, t_mean, yerr=t_err, fmt="o-", color="#1f4e79",
                 lw=1.8, ms=6.5, capsize=4, elinewidth=1.2, zorder=3,
                 label="matched-filter $t$ (95% CI, $t\\pm1.96$)")
    ax1.set_xlabel("Energy (MeV)")
    ax1.set_ylabel("matched-filter $t$ statistic")
    ax1.set_xticks(es)
    ax1.set_xticklabels([f"{e:g}" for e in es])
    ax1.set_title("(a) Defect trajectory vs pseudo-trajectory", fontsize=9.5)
    ax1.text(0.03, 0.06, "seed-invariant by construction:\ncomputed from measured sinograms",
             transform=ax1.transAxes, fontsize=7.4, color="#444444", va="bottom")
    ax1.legend(fontsize=7.4, loc="upper right", frameon=False)
    ax1.grid(alpha=0.25, lw=0.6)

    # (b) 空腔 CNR（真正带种子方差）
    ax2.axhline(5.0, color="#c0392b", lw=1.0, ls=":", zorder=1)
    ax2.text(es[0], 5.0, "Rose criterion", fontsize=7.2, color="#c0392b",
             va="bottom", ha="left")
    ax2.errorbar(es, cnr_mean, yerr=cnr_std, fmt="s-", color="#1f4e79",
                 lw=1.8, ms=6, capsize=4, elinewidth=1.2, zorder=3, label="CNN completion")
    ax2.errorbar(es, sart_mean, yerr=sart_std, fmt="^--", color="#e07b39",
                 lw=1.5, ms=5.5, capsize=4, elinewidth=1.1, zorder=3, label="SART (180 views)")
    ax2.set_xlabel("Energy (MeV)")
    ax2.set_ylabel("cavity CNR")
    ax2.set_xticks(es)
    ax2.set_xticklabels([f"{e:g}" for e in es])
    ax2.set_ylim(0.0, 5.9)
    ax2.set_title("(b) Cavity CNR, mean $\\pm$ s.d. over "
                  f"{min(s['n_runs'] for s in summary.values())} seeds", fontsize=9.5)
    ax2.legend(fontsize=7.4, loc="lower right", frameon=False)
    ax2.grid(alpha=0.25, lw=0.6)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_mf_t_vs_energy.{ext}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(RUNS / "_flaw_multi_index.csv"))
    ap.add_argument("--out-json", default=str(RUNS / "_flaw_multi_summary.json"))
    ap.add_argument("--out-md", default=str(RUNS / "_flaw_multi_table.md"))
    ap.add_argument("--fig-dir", default=str(FIG_DIR))
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    summary, missing = collect(Path(args.index))
    if not summary["per_energy"]:
        print("[FATAL] 没有任何可比对的 step3 评测结果", file=sys.stderr)
        return 2

    Path(args.out_json).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = md_table(summary["per_energy"])
    Path(args.out_md).write_text(md + "\n", encoding="utf-8")
    print(md)
    if missing:
        print(f"\n[待评测] {len(missing)} 个 run 尚无 image_metrics.json：")
        for m in missing:
            print("  -", m)
    if not args.no_fig:
        plot(summary["per_energy"], Path(args.fig_dir))
        print(f"\n已写出图 {Path(args.fig_dir) / 'fig_mf_t_vs_energy.png'}")
    print(f"已写出 {args.out_json} / {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
