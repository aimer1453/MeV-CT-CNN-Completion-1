"""新旧两批缺陷数据的并排对照（Table 8 换代用）。

输入：两份 aggregate 产物
  旧（无包壳，3_3_flaw）      runs/_flaw_multi_summary.json
  新（带包壳，3_3_flaw_1_1）  runs/_flaw11_multi_summary.json
输出：屏幕表 + Markdown（可直接粘进交接文档）

为什么需要这个脚本
------------------
Table 8 的四档数字要整体换代。换代时必须能一眼看出**哪些量的方向变了**，
而不是只看新数字好不好看。这里固定输出 6 个量 + 4 个方法对照，逐能量打 Δ%。

用法
----
  python scripts/flaw11_compare_old.py
  python scripts/flaw11_compare_old.py --out runs/_flaw11_vs_old.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

OLD = RUNS / "_flaw_multi_summary.json"
NEW = RUNS / "_flaw11_multi_summary.json"

METHODS = ("cnn_completed", "linear_interp", "fbp_sparse", "sart_sparse", "dense")


def rel(new: float, old: float) -> str:
    if old in (None, 0) or new is None:
        return "—"
    return f"{100.0 * (new - old) / abs(old):+.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=str(OLD))
    ap.add_argument("--new", default=str(NEW))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    for p in (Path(args.old), Path(args.new)):
        if not p.is_file():
            raise SystemExit(f"[FATAL] 缺 {p}（先跑 flaw_multi_aggregate.py）")
    old = json.loads(Path(args.old).read_text(encoding="utf-8"))["per_energy"]
    new = json.loads(Path(args.new).read_text(encoding="utf-8"))["per_energy"]
    es = sorted(set(old) & set(new), key=float)
    if len(es) != 4:
        print(f"[WARN] 能量档不齐，交集只有 {es}")

    L: list[str] = []
    L.append("### 4.1 Table 8 主行：CNN 补全的图像域与可检测性（n=3）")
    L.append("")
    L.append("| 能量 | 批 | RMSE_valid ($10^{-3}$) | PSNR$_c$ (dB) | SSIM | CNR (CNN) | CNR (dense) | $t$ |")
    L.append("|---|---|---|---|---|---|---|---|")
    for e in es:
        o, n = old[e], new[e]
        for tag, s in (("旧（无包壳）", o), ("新（带包壳）", n)):
            c = s["image"]["cnn_completed"]
            L.append(f"| {e} | {tag} | {c['rmse_valid']['mean']*1e3:.3f} ± "
                     f"{c['rmse_valid']['std']*1e3:.3f} | "
                     f"{s['psnr_common']['cnn_completed']['mean']:.2f} | "
                     f"{c['ssim']['mean']:.4f} ± {c['ssim']['std']:.4f} | "
                     f"{s['cnr']['cnn_completed']['mean']:.2f} ± "
                     f"{s['cnr']['cnn_completed']['std']:.2f} | "
                     f"{s['cnr']['dense']['mean']:.2f} | "
                     f"{s['mf_t_stat']['mean']:.2f} |")
        c_o, c_n = o["image"]["cnn_completed"], n["image"]["cnn_completed"]
        L.append(f"| {e} | **Δ** | {rel(c_n['rmse_valid']['mean'], c_o['rmse_valid']['mean'])} | "
                 f"{rel(n['psnr_common']['cnn_completed']['mean'], o['psnr_common']['cnn_completed']['mean'])} | "
                 f"{rel(c_n['ssim']['mean'], c_o['ssim']['mean'])} | "
                 f"{rel(n['cnr']['cnn_completed']['mean'], o['cnr']['cnn_completed']['mean'])} | "
                 f"{rel(n['cnr']['dense']['mean'], o['cnr']['dense']['mean'])} | "
                 f"{rel(n['mf_t_stat']['mean'], o['mf_t_stat']['mean'])} |")
    L.append("")

    L.append("### 4.2 图像域 RMSE 的方法排序（$10^{-3}$，n=3 均值）")
    L.append("")
    L.append("| 能量 | 批 | CNN | linear | FBP 稀疏 | SART 稀疏 |")
    L.append("|---|---|---|---|---|---|")
    for e in es:
        for tag, s in (("旧", old[e]), ("新", new[e])):
            cells = []
            for m in ("cnn_completed", "linear_interp", "fbp_sparse", "sart_sparse"):
                v = s["image"][m]["rmse_valid"]
                cells.append(f"{v['mean']*1e3:.3f} ± {v['std']*1e3:.3f}")
            L.append(f"| {e} | {tag} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("### 4.3 匹配滤波的分解（ΔL / SEM → t）")
    L.append("")
    L.append("| 能量 | 批 | ΔL | SEM | $t$ | 伪轨迹基线 | 伪轨迹极差 |")
    L.append("|---|---|---|---|---|---|---|")
    for e in es:
        for tag, s in (("旧", old[e]), ("新", new[e])):
            mf = s["mf_t_stat"]
            L.append(f"| {e} | {tag} | {s.get('mf_contrast_mean', float('nan')):+.5f} | "
                     f"{s.get('mf_sem_mean', float('nan')):.5f} | {mf['mean']:.2f} | — | — |")
    L.append("")

    # ---- 方向一致性检查：哪些量的排序在两批之间反转了
    L.append("### 4.4 换代是否改变结论（自动检查）")
    L.append("")
    checks = []
    for e in es:
        for key, get in (
            ("CNN < linear（RMSE）",
             lambda s: s["image"]["cnn_completed"]["rmse_valid"]["mean"]
             < s["image"]["linear_interp"]["rmse_valid"]["mean"]),
            ("CNN < SART（RMSE）",
             lambda s: s["image"]["cnn_completed"]["rmse_valid"]["mean"]
             < s["image"]["sart_sparse"]["rmse_valid"]["mean"]),
            ("CNN > FBP 稀疏（RMSE 更差）",
             lambda s: s["image"]["cnn_completed"]["rmse_valid"]["mean"]
             > s["image"]["fbp_sparse"]["rmse_valid"]["mean"]),
            ("SART CNR 最高",
             lambda s: max(s["cnr"], key=lambda m: (s["cnr"][m]["mean"] or -1)) == "sart_sparse"),
            ("CNN CNR < 5（Rose）",
             lambda s: (s["cnr"]["cnn_completed"]["mean"] or 0) < 5),
            ("dense CNR < 5（Rose）",
             lambda s: (s["cnr"]["dense"]["mean"] or 0) < 5),
        ):
            a, b = bool(get(old[e])), bool(get(new[e]))
            checks.append((e, key, a, b))
    L.append("| 能量 | 判据 | 旧批 | 新批 | 一致 |")
    L.append("|---|---|---|---|---|")
    for e, key, a, b in checks:
        L.append(f"| {e} | {key} | {'是' if a else '否'} | {'是' if b else '否'} | "
                 f"{'✓' if a == b else '**✗ 反转**'} |")
    n_flip = sum(1 for _e, _k, a, b in checks if a != b)
    L.append("")
    L.append(f"反转条数：**{n_flip} / {len(checks)}**")

    text = "\n".join(L)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
