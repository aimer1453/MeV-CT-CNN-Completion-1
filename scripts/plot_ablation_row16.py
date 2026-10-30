"""两视角 vs 三视角消融对比图（row16 主指标）。

数据来源（只读，不重算）：
  runs/_ablation_3view_baseline.json                   3 视角（0400 run 的 recon）
  runs/<rid>_cnn_recon/test_metrics.json               2 视角（0052 run 的 recon）

输出：docs/figs/p01_ablation_2view/fig_ablation_row16.{png,pdf}
图内容：逐能量 row16 RMSE（A/B 相位平均）分组柱 + 相对差标注；
        脚注写出绝对差（-log 域）以说明"相对差随能量放大是因为基线缩小"。

用法：python scripts/plot_ablation_row16.py <2view_run_id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT_DIR = ROOT / "docs" / "figs" / "p01_ablation_2view"
ENERGIES = ["1.5", "3", "4.5", "6"]
C3, C2 = "#3B6EA5", "#D0783A"      # 3 视角 / 2 视角


def load(rid2: str) -> tuple[dict, dict]:
    base = json.loads((RUNS / "_ablation_3view_baseline.json").read_text(encoding="utf-8"))
    m2 = json.loads((RUNS / f"{rid2}_cnn_recon" / "test_metrics.json")
                    .read_text(encoding="utf-8"))
    two: dict[str, dict] = {}
    for e, d in m2["projection_domain_check"].items():
        for ph in ("A", "B"):
            two[f"{e}_{ph}"] = {"row16_all": d["by_input_phase"][ph]["row16"]["rmse_valid_all"]}
    return base["projection_domain"], two


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    rid2 = sys.argv[1]
    b, t = load(rid2)
    mean3 = [sum(b[f"{e}_{p}"]["row16_all"] for p in "AB") / 2 for e in ENERGIES]
    mean2 = [sum(t[f"{e}_{p}"]["row16_all"] for p in "AB") / 2 for e in ENERGIES]
    d_rel = [100 * (y - x) / x for x, y in zip(mean3, mean2)]
    d_abs = [y - x for x, y in zip(mean3, mean2)]

    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=200)
    xs = range(len(ENERGIES))
    w = 0.36
    b1 = ax.bar([x - w / 2 for x in xs], mean3, w, color=C3, label="3 input views (main text)")
    b2 = ax.bar([x + w / 2 for x in xs], mean2, w, color=C2, label="2 input views (ablation)")
    for rect, v in zip(b1, mean3):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", va="bottom", fontsize=7, color=C3)
    for rect, v, dr in zip(b2, mean2, d_rel):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", va="bottom", fontsize=7, color=C2)
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.013, f"+{dr:.1f}%",
                ha="center", va="bottom", fontsize=7.5, color="#8C3B12", fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{e} MeV" for e in ENERGIES])
    ax.set_ylabel("row16 RMSE (-log domain, valid pixels)")
    ax.set_ylim(0, max(mean2) * 1.28)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Sparse-view completion: dropping the $\\theta-3^{\\circ}$ view",
                 fontsize=9.5)
    fig.text(0.5, 0.005,
             "Bars: mean over input phases A/B. Absolute gap stays near "
             f"{sum(d_abs) / len(d_abs):.4f} (-log units) across energies; the growing "
             "relative gap reflects the shrinking baseline.",
             ha="center", fontsize=6.8, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_ablation_row16.{ext}")
    plt.close(fig)
    print(f"已生成 {OUT_DIR}/fig_ablation_row16.{{png,pdf}}")
    for e, x, y, dr, da in zip(ENERGIES, mean3, mean2, d_rel, d_abs):
        print(f"  {e:>4} MeV: 3v={x:.6f} 2v={y:.6f} Δ={dr:+.1f}% (abs {da:+.5f})")


if __name__ == "__main__":
    main()
