#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_100ep_composite.py —— 100-epoch 训练曲线合成图（fig_100ep_curves.png）

背景：审稿人要求主文放 100-epoch 曲线（原 Fig 3/Fig 4/Fig 5 是 5-epoch pilot
      20261022_2352）。本图把三联面板压进一张：
        (a) 训练 loss（逐能量）   (b) 验证 loss（逐能量）   (c) 验证 RMSE（逐能量）
      并在 (c) 上标出 best epoch（验证集物理域 RMSE 最低的 checkpoint）。

输入（只读中间数据，不读 npy）：
    runs/<run_id>/train_log.csv（14 列 schema，同 plot_training_curves.py）

输出：
    docs/figs/p01_fig345_100ep/fig_100ep_curves.png（200 dpi，bbox tight）
    docs/figs/p01_fig345_100ep/fig_100ep_curves.pdf（矢量，论文侧备用）

用法：
    conda run -n ct_full python scripts/plot_100ep_composite.py \
        --run-id 20261024_0400_cnn_bs8_lr1e-4
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
OUT_DIR = ROOT / "docs" / "figs" / "p01_fig345_100ep"

ENERGIES: tuple[str, ...] = ("1.5", "3", "4.5", "6")
GRAYS: tuple[str, ...] = ("#000000", "#555555", "#888888", "#bbbbbb")
STYLES: tuple[str, ...] = ("-", "--", "-.", ":")
MARKERS: tuple[str, ...] = ("o", "s", "^", "D")

REQUIRED_COLUMNS: tuple[str, ...] = (
    "epoch", "split", "energy_MeV", "n_samples", "coverage", "valid_coverage",
    "rmse_valid", "rmse_full", "psnr", "ssim", "n_invariant_warn", "lr",
    "loss_train", "loss_val",
)


class PlotSpecError(RuntimeError):
    """输入数据不合规时抛出（fail fast，不做静默兜底）。"""


def load_log(run_id: str) -> list[dict]:
    path = RUNS_DIR / run_id / "train_log.csv"
    if not path.is_file():
        raise PlotSpecError(f"缺少 {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = tuple(reader.fieldnames or ())
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise PlotSpecError(f"{path.name} 缺列: {missing}")
        rows: list[dict] = []
        for r in reader:
            try:
                rows.append({
                    "epoch": int(r["epoch"]),
                    "split": r["split"],
                    "energy_MeV": r["energy_MeV"],
                    "n_samples": int(r["n_samples"]),
                    "rmse_valid": float(r["rmse_valid"]),
                    "loss_train": float(r["loss_train"]),
                    "loss_val": float(r["loss_val"]),
                })
            except (KeyError, ValueError) as exc:
                raise PlotSpecError(f"{path.name} 行解析失败: {r} ({exc})") from exc
    if not rows:
        raise PlotSpecError(f"{path.name} 没有数据行")
    return rows


def series(rows: list[dict], split: str, energy: str, key: str):
    sel = sorted((r for r in rows if r["split"] == split and r["energy_MeV"] == energy),
                 key=lambda r: r["epoch"])
    return [r["epoch"] for r in sel], [r[key] for r in sel]


def epoch_ticks(epochs: list[int]) -> list[int]:
    if len(epochs) <= 12:
        return epochs
    step = max(1, round(len(epochs) / 10))
    ticks = list(range(epochs[0], epochs[-1] + 1, step))
    if ticks[-1] != epochs[-1]:
        ticks.append(epochs[-1])
    return ticks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="100-epoch 三联合成图（读 train_log.csv）")
    ap.add_argument("--run-id", required=True, help="runs/ 下的目录名")
    args = ap.parse_args(argv)

    try:
        rows = load_log(args.run_id)
    except PlotSpecError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    epochs = sorted({r["epoch"] for r in rows})
    n_tr = sum(r["n_samples"] for r in rows if r["split"] == "train" and r["epoch"] == epochs[0])
    n_va = sum(r["n_samples"] for r in rows if r["split"] == "val" and r["epoch"] == epochs[0])
    per_tr = sorted({r["n_samples"] for r in rows if r["split"] == "train" and r["epoch"] == epochs[0]})
    per_va = sorted({r["n_samples"] for r in rows if r["split"] == "val" and r["epoch"] == epochs[0]})

    # best epoch = 四档验证集物理域 RMSE 各自取最低后多数一致的那个 epoch
    best_votes: dict[int, int] = {}
    best_rmse: dict[str, tuple[int, float]] = {}
    for e in ENERGIES:
        xs, ys = series(rows, "val", e, "rmse_valid")
        b = min(zip(xs, ys), key=lambda t: t[1])
        best_rmse[e] = (b[0], b[1])
        best_votes[b[0]] = best_votes.get(b[0], 0) + 1
    best_epoch = max(best_votes, key=lambda k: best_votes[k])

    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "axes.edgecolor": "black",
    })

    # 画布按正文 \textwidth ≈ 6.3 in 设计：8.9 in → 缩放 0.71，10 pt 落到正文 ≈7.1 pt
    fig = plt.figure(figsize=(8.9, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.24)
    ax_tr = fig.add_subplot(gs[0, 0])
    ax_va = fig.add_subplot(gs[0, 1])
    ax_rm = fig.add_subplot(gs[1, :])

    for ax, split, key, title in (
        (ax_tr, "train", "loss_train", "Training Loss (normalized)"),
        (ax_va, "val", "loss_val", "Validation Loss (normalized)"),
    ):
        for i, e in enumerate(ENERGIES):
            xs, ys = series(rows, split, e, key)
            ax.plot(xs, ys, color=GRAYS[i], ls=STYLES[i], marker=MARKERS[i],
                    ms=4.2, lw=1.4, markevery=max(1, len(xs) // 20),
                    label=f"{e} MeV (last {ys[-1]:.4f})")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Masked MSE (normalized)")
        ax.set_xticks(epoch_ticks(epochs))
        ax.set_xlim(epochs[0] - 0.18, epochs[-1] + 0.25)
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(y_lo, y_hi + 0.38 * (y_hi - y_lo))
        ax.legend(fontsize=7.6, loc="upper right", framealpha=0.95,
                  labelspacing=0.32, borderpad=0.38, handlelength=1.7)

    for i, e in enumerate(ENERGIES):
        xs, ys = series(rows, "val", e, "rmse_valid")
        ax_rm.plot(xs, ys, color=GRAYS[i], ls=STYLES[i], marker=MARKERS[i],
                   ms=4.6, lw=1.5, markevery=max(1, len(xs) // 20),
                   label=f"{e} MeV (last {ys[-1]:.4f})")
    ax_rm.axvline(best_epoch, color="#000000", lw=1.0, ls=(0, (2, 2)), alpha=0.65)
    # 标注放虚线顶端左侧：该区域（x<best_epoch, y 接近顶部）无曲线经过；
    # 放底端会贴着 6 MeV 曲线末段，放右侧会与图例冲突。
    ax_rm.annotate(f"best checkpoint, epoch {best_epoch}",
                   xy=(best_epoch, 1.0), xycoords=("data", "axes fraction"),
                   xytext=(-5, -8), textcoords="offset points",
                   fontsize=7.8, color="#333333", ha="right", va="top")
    ax_rm.set_title("Validation RMSE vs Epoch by Energy (physical domain)", fontweight="bold")
    ax_rm.set_xlabel("Epoch")
    ax_rm.set_ylabel("Validation RMSE (physical domain)")
    ax_rm.set_xticks(epoch_ticks(epochs))
    ys_all = [r["rmse_valid"] for r in rows if r["split"] == "val"]
    y_lo, y_hi = min(ys_all), max(ys_all)
    span = max(y_hi - y_lo, 1e-6)
    ax_rm.set_ylim(y_lo - 0.10 * span, y_hi + 0.42 * span)
    ax_rm.set_xlim(epochs[0] - 0.18, epochs[-1] + 0.25)
    # 图例放左上：顶部留白带（y > 曲线起点 0.91）全宽空闲，且避开 x=best_epoch 处的虚线
    ax_rm.legend(fontsize=7.6, loc="upper left", framealpha=0.95,
                 labelspacing=0.32, borderpad=0.38, handlelength=1.7)

    fig.suptitle("100-Epoch Training Curves by Energy (masked MSE; zero-count pixels excluded)",
                 fontweight="bold", fontsize=11.5, y=0.995)
    footer = (f"run {args.run_id} · batch 8 · lr 1e-4 · warmup 50 steps · "
              f"train n = {n_tr} ({per_tr[0]} per energy) · val n = {n_va} ({per_va[0]} per energy)\n"
              f"loss: per-batch masked MSE (normalised line-integral domain) · "
              f"RMSE: valid pixels only (I > 0 & empty_proj > 0) · best epoch {best_epoch}")
    t = fig.text(0.5, -0.005, footer, ha="center", va="top", fontsize=8.0,
                 color="#555555", linespacing=1.45)
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    if bb.width > fig.bbox.width * 0.995:
        raise PlotSpecError(f"图底元数据超宽：{bb.width:.0f}px > {fig.bbox.width:.0f}px")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "fig_100ep_curves.png"
    pdf = OUT_DIR / "fig_100ep_curves.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    for p in (png, pdf):
        if not p.is_file() or p.stat().st_size == 0:
            print(f"[FAIL] 图未生成或为空: {p}", file=sys.stderr)
            return 4
        print(f"已生成 {p}  ({p.stat().st_size / 1024:.1f} KB)")
    print(f"best epoch = {best_epoch}（四档投票 {best_votes}）")
    for e in ENERGIES:
        print(f"  {e:>4s} MeV: best epoch {best_rmse[e][0]}, val rmse {best_rmse[e][1]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
