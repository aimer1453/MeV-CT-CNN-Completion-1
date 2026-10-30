#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_training_curves.py —— 训练过程图（3 张）

输入（只读中间数据，**不读原始 npy**）：
    runs/<run_id>/train_log.csv     （14 列，由 train.py 产出）

输出（每张图都放 docs/figs/<run_id>/，以便入库）：
    docs/figs/<run_id>/train_loss_curve_<run_id>.png      分能量 train / val loss 随 epoch
    docs/figs/<run_id>/val_rmse_by_energy_<run_id>.png    分能量 val_rmse_valid 随 epoch
    docs/figs/<run_id>/coverage_by_energy_<run_id>.png    分档 coverage / valid_coverage 双柱

约定（见 docs/figures_and_reports_spec.md）：
  - 只用 matplotlib，不依赖 seaborn / plotly
  - 灰阶 + 线型 + marker 区分，黑白可打印
  - 每图必备：X 轴标签（含单位）、Y 轴标签（含单位）、图例、关键数值标注
  - **图内不写标题，也不写图底灰注**（R65 起）：标题与说明归 \caption 和正文，
    run id / batch / lr / warmup / 样本量归 `docs/figs/<run_id>/README.md`；
    图内只保留随图例走的末轮数值与柱顶数值标签
  - DPI >= 200，bbox_inches="tight"
  - 中文字体 Microsoft YaHei / SimHei，回退 DejaVu Sans，axes.unicode_minus=False
  - 读数据先校验字段完整性，缺字段直接报错退出
  - 生成后打印每张图路径与文件大小；任一失败退出码非 0

用法：
  conda run -n ct_full python scripts/plot_training_curves.py
  conda run -n ct_full python scripts/plot_training_curves.py --run-id 20261022_2352_cnn_bs8_lr1e-4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
FIGS_DIR = ROOT / "docs" / "figs"

ENERGIES: tuple[str, ...] = ("1.5", "3", "4.5", "6")
# Okabe–Ito 4 能量配色（1.5/3/4.5/6 MeV，冷→暖，色盲友好，白底去黑）；
# 同时保留 STYLES + MARKERS 作冗余，黑白打印仍可区分。
ENERGY_COLORS: tuple[str, ...] = ("#0072B2", "#009E73", "#E69F00", "#D55E00")
STYLES: tuple[str, ...] = ("-", "--", "-.", ":")
MARKERS: tuple[str, ...] = ("o", "s", "^", "D")

#: 双语输出：zh -> _analysis.png（自己看），en -> _paper.png（论文用）
LANGS: tuple[str, ...] = ("zh", "en")
SUFFIX: dict[str, str] = {"zh": "analysis", "en": "paper"}
FONTS: dict[str, list[str]] = {
    "zh": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "en": ["DejaVu Sans", "Arial"],          # 英文版禁止挂中文字体，否则会回退
}

#: 图上文案。中英一一对应。**只留轴标签与图例**——标题/灰注已删（R65），
#: 故这里不再有 title_* / cap_* 键，避免"删了调用却留下文案"的半成品状态。
T: dict[str, dict[str, str]] = {
    "zh": {
        "x_epoch": "epoch（轮次，无量纲）",
        "y_loss": "Masked MSE（归一化域，无量纲）",
        "y_rmse": "val_rmse_valid（物理域 -log 值，无量纲）",
        "x_energy": "能量档 (MeV)",
        "y_coverage": "覆盖率（0-1，无量纲）",
        "leg_energy": "{e} MeV",
        "leg_last": "{e} MeV（末轮 {v:.4f}）",
        "leg_coverage": "coverage（资产口径）",
        "leg_valid_coverage": "valid_coverage（评估口径）",
    },
    "en": {
        "x_epoch": "Epoch",
        "y_loss": "Masked MSE (normalized)",
        "y_rmse": "Validation RMSE (physical domain)",
        "x_energy": "Energy (MeV)",
        "y_coverage": "Coverage (0-1)",
        "leg_energy": "{e} MeV",
        "leg_last": "{e} MeV (last {v:.4f})",
        "leg_coverage": "Coverage (asset scope)",
        "leg_valid_coverage": "Valid Coverage (eval scope)",
    },
}


#: train_log.csv 必须包含的列（顺序即 schema，缺一即报错）
REQUIRED_COLUMNS: tuple[str, ...] = (
    "epoch", "split", "energy_MeV", "n_samples", "coverage", "valid_coverage",
    "rmse_valid", "rmse_full", "psnr", "ssim", "n_invariant_warn", "lr",
    "loss_train", "loss_val",
)


class PlotSpecError(RuntimeError):
    """输入数据不合规时抛出（fail fast，不做静默兜底）。"""


# ------------------------------------------------------------------ 基础设施
def setup_style(lang: str) -> None:
    """按语言切换字体。英文版只挂 DejaVu/Arial，杜绝中文回退。"""
    if lang not in LANGS:
        raise PlotSpecError(f"未知 lang: {lang!r}，应为 {LANGS}")
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONTS[lang],
        "axes.unicode_minus": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def pick_run_id(arg: str | None) -> str:
    """未指定时取 runs/ 下最新一个含 train_log.csv 的目录。"""
    if arg:
        d = RUNS_DIR / arg
        if not (d / "train_log.csv").is_file():
            raise PlotSpecError(f"找不到 {d / 'train_log.csv'}")
        return arg
    cands = [p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "train_log.csv").is_file()]
    if not cands:
        raise PlotSpecError(f"{RUNS_DIR} 下没有任何含 train_log.csv 的 run 目录")
    return max(cands, key=lambda p: p.name).name


def load_log(run_id: str) -> list[dict]:
    """读 train_log.csv 并做字段完整性 + 数值有效性校验。"""
    path = RUNS_DIR / run_id / "train_log.csv"
    if not path.is_file():
        raise PlotSpecError(f"缺少 {path}")

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = tuple(reader.fieldnames or ())
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise PlotSpecError(f"{path.name} 缺列: {missing}（应为 14 列 schema）")
        extra = [c for c in header if c not in REQUIRED_COLUMNS]
        if extra:
            raise PlotSpecError(f"{path.name} 含未定义列: {extra}")

        rows: list[dict] = []
        for r in reader:
            try:
                row = {
                    "epoch": int(r["epoch"]),
                    "split": r["split"],
                    "energy_MeV": r["energy_MeV"],
                    "n_samples": int(r["n_samples"]),
                    "coverage": float(r["coverage"]),
                    "valid_coverage": float(r["valid_coverage"]),
                    "rmse_valid": float(r["rmse_valid"]),
                    "rmse_full": float(r["rmse_full"]),
                    "n_invariant_warn": int(r["n_invariant_warn"]),
                    "lr": float(r["lr"]),
                    "loss_train": float(r["loss_train"]),
                    "loss_val": float(r["loss_val"]),
                }
            except (KeyError, ValueError) as exc:
                raise PlotSpecError(f"{path.name} 行解析失败: {r}  ({exc})") from exc
            if row["split"] not in ("train", "val", "test"):
                raise PlotSpecError(f"未知 split: {row['split']!r}")
            if row["energy_MeV"] not in ENERGIES:
                raise PlotSpecError(f"未知能量: {row['energy_MeV']!r}，应在 {ENERGIES}")
            rows.append(row)

    if not rows:
        raise PlotSpecError(f"{path.name} 没有数据行")

    epochs = sorted({r["epoch"] for r in rows})
    if len(epochs) < 2:
        raise PlotSpecError(f"只有 {len(epochs)} 个 epoch，无法画趋势线")

    # 每个 (split, energy) 都必须在所有 epoch 上齐全
    for split in ("train", "val"):
        for e in ENERGIES:
            got = sorted(r["epoch"] for r in rows if r["split"] == split and r["energy_MeV"] == e)
            if got and got != epochs:
                raise PlotSpecError(
                    f"split={split} energy={e} 的 epoch 不完整: {got} vs {epochs}"
                )
    return rows


def epoch_ticks(epochs: list[int]) -> list[int]:
    """x 轴刻度抽稀。

    5-epoch pilot 逐 epoch 打刻度无妨；100 epoch 时每轮一个标签会互相重叠
    糊成一片（审稿人曾据此判"不可读"）。按 ~10 个主刻度抽稀，末轮必含。
    """
    if len(epochs) <= 12:
        return epochs
    step = max(1, round(len(epochs) / 10))
    ticks = list(range(epochs[0], epochs[-1] + 1, step))
    if ticks[-1] != epochs[-1]:
        ticks.append(epochs[-1])
    return ticks


def series(rows: list[dict], split: str, energy: str, key: str) -> tuple[list[int], list[float]]:
    sel = sorted(
        (r for r in rows if r["split"] == split and r["energy_MeV"] == energy),
        key=lambda r: r["epoch"],
    )
    return [r["epoch"] for r in sel], [r[key] for r in sel]


def all_energy_mean(rows: list[dict], split: str, key: str) -> float:
    """ALL = 按 n_samples 加权平均（4 档样本数相同则等价算术平均）。"""
    sel = [r for r in rows if r["split"] == split]
    if not sel:
        raise PlotSpecError(f"split={split} 无数据，无法计算 ALL")
    w = sum(r["n_samples"] for r in sel)
    return sum(r[key] * r["n_samples"] for r in sel) / w if w else float("nan")


def split_counts(rows: list[dict], epoch: int) -> dict[str, object]:
    """取某 epoch 的 train/val 样本总数与每档分量。

    ⚠ R65 起图内不再印样本量，本函数在绘图路径里已无调用点；保留是因为
    `docs/figs/<run_id>/README.md` 的 n = 1080（270/档）/ 180（45/档）需要用
    同一份 CSV 复核，手工核对时可 import 直接用，避免再写一遍分组逻辑。
    各档分量不一致时用 "a/b/..." 如实呈现，不取平均、不隐瞒。
    """
    out: dict[str, object] = {}
    for split, short in (("train", "tr"), ("val", "va")):
        sel = [r for r in rows if r["split"] == split and r["epoch"] == epoch]
        if not sel:
            raise PlotSpecError(f"split={split} 在 epoch {epoch} 无数据")
        out[f"n_{short}"] = sum(r["n_samples"] for r in sel)
        per = sorted({r["n_samples"] for r in sel})
        out[f"n_{short}_e"] = per[0] if len(per) == 1 else "/".join(str(v) for v in per)
    return out


def _save(fig, out: Path) -> Path:
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------ 图 1
def fig_loss_curve(rows: list[dict], run_id: str, lang: str) -> Path:
    t = T[lang]
    epochs = sorted({r["epoch"] for r in rows})
    # 画布按"正文实际尺寸"设计：\textwidth ≈ 6.3 in，故取 8.4 in → 缩放 0.75，
    # 10 pt 基础字号落到正文 ≈7.5 pt（旧版 12.4 in 缩放 0.51，字只剩 5 pt，被审稿人判为不可读）。
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.2), sharex=True)

    # 面板标题已删（R65）：图内不写标题，两张并排靠 X/Y 轴标签区分，
    # 各自的含义由 \caption 说明（= loss 分别在 train / val split 上）。
    for ax, split in ((axes[0], "train"), (axes[1], "val")):
        for i, e in enumerate(ENERGIES):
            xs, ys = series(rows, split, e, "loss_train" if split == "train" else "loss_val")
            # 末轮数值并入图例，不再在曲线末端外挂数字：末端标注会贴着右边框，
            # 低值两条曲线（0.1972 / 0.1100）互相挤压，缩放后即成糊字。
            ax.plot(xs, ys, color=ENERGY_COLORS[i], ls=STYLES[i], marker=MARKERS[i],
                    ms=4.2, lw=1.4,
                    markevery=max(1, len(xs) // 20),
                    label=t["leg_last"].format(e=e, v=ys[-1]))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel(t["x_epoch"])
        ax.set_ylabel(t["y_loss"])
        ax.set_xticks(epoch_ticks(epochs))
        ax.set_xlim(epochs[0] - 0.18, epochs[-1] + 0.25)
        y_lo, y_hi = ax.get_ylim()
        # 顶部留白必须够放 4 行带数值的图例：留白不足时图例会压住 1.5 MeV 曲线右段
        ax.set_ylim(y_lo, y_hi + 0.38 * (y_hi - y_lo))
        ax.legend(fontsize=7.8, loc="upper right", framealpha=0.95,
                  labelspacing=0.32, borderpad=0.38, handlelength=1.7)

    fig.tight_layout()
    out = FIGS_DIR / run_id / f"train_loss_curve_{run_id}_{SUFFIX[lang]}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    return _save(fig, out)


# ------------------------------------------------------------------ 图 2
def fig_val_rmse(rows: list[dict], run_id: str, lang: str) -> Path:
    t = T[lang]
    epochs = sorted({r["epoch"] for r in rows})
    # 0.85\textwidth ≈ 5.35 in；画布取 7.4 in → 缩放 0.72，字不再被压成 4 pt。
    fig, ax = plt.subplots(figsize=(7.4, 3.9))

    for i, e in enumerate(ENERGIES):
        xs, ys = series(rows, "val", e, "rmse_valid")
        # 末轮数值已写进图例，末端外挂标注是重复信息且会顶到右边框，故删除。
        ax.plot(xs, ys, color=ENERGY_COLORS[i], ls=STYLES[i], marker=MARKERS[i],
                ms=4.6, lw=1.5,
                markevery=max(1, len(xs) // 20),
                label=t["leg_last"].format(e=e, v=ys[-1]))

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel(t["x_epoch"])
    ax.set_ylabel(t["y_rmse"])
    ax.set_xticks(epoch_ticks(epochs))
    # 曲线整体单调下降，右上角为唯一空白区；抬高 ylim 顶部让图例落在 1.5 MeV 起点上方
    ys_all = [r["rmse_valid"] for r in rows if r["split"] == "val"]
    y_lo, y_hi = min(ys_all), max(ys_all)
    span = max(y_hi - y_lo, 1e-6)
    ax.set_ylim(y_lo - 0.10 * span, y_hi + 0.42 * span)
    ax.set_xlim(epochs[0] - 0.18, epochs[-1] + 0.25)
    ax.legend(fontsize=7.8, loc="upper right", framealpha=0.95,
              labelspacing=0.32, borderpad=0.38, handlelength=1.7)

    fig.tight_layout()
    out = FIGS_DIR / run_id / f"val_rmse_by_energy_{run_id}_{SUFFIX[lang]}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    return _save(fig, out)


# ------------------------------------------------------------------ 图 3
def fig_coverage(rows: list[dict], run_id: str, lang: str) -> Path:
    t = T[lang]
    labels = list(ENERGIES) + ["ALL"]
    cov, vcov = [], []
    for e in labels:
        if e == "ALL":
            cov.append(all_energy_mean(rows, "train", "coverage"))
            vcov.append(all_energy_mean(rows, "train", "valid_coverage"))
        else:
            first = min((r for r in rows if r["split"] == "train" and r["energy_MeV"] == e),
                        key=lambda r: r["epoch"])
            cov.append(first["coverage"])
            vcov.append(first["valid_coverage"])

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    b1 = ax.bar(x - w / 2, cov, w, color="#0072B2", edgecolor="black", lw=0.6,
                label=t["leg_coverage"])
    b2 = ax.bar(x + w / 2, vcov, w, color="#56B4E9", edgecolor="black", lw=0.6, hatch="///",
                label=t["leg_valid_coverage"])

    for rect, v in list(zip(b1, cov)) + list(zip(b2, vcov)):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.4f}",
                ha="center", va="bottom", fontsize=7.8)

    ax.set_ylim(0, 1.22)
    ax.set_xticks(x, [f"{l} MeV" if l != "ALL" else "ALL" for l in labels])
    ax.set_xlabel(t["x_energy"])
    ax.set_ylabel(t["y_coverage"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.4, loc="upper center", framealpha=0.95, borderpad=0.5)
    fig.tight_layout()
    out = FIGS_DIR / run_id / f"coverage_by_energy_{run_id}_{SUFFIX[lang]}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    return _save(fig, out)


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="训练过程图（读 train_log.csv，不读 npy；双语双份输出）")
    ap.add_argument("--run-id", default=None, help="runs/ 下的目录名；缺省取最新")
    ap.add_argument("--lang", choices=LANGS, default=None,
                    help="只出某一语言；缺省两种都出（analysis=zh, paper=en）")
    args = ap.parse_args(argv)

    langs = (args.lang,) if args.lang else LANGS
    try:
        run_id = pick_run_id(args.run_id)
        rows = load_log(run_id)
        outs = []
        for lang in langs:
            setup_style(lang)
            outs += [
                fig_loss_curve(rows, run_id, lang),
                fig_val_rmse(rows, run_id, lang),
                fig_coverage(rows, run_id, lang),
            ]
    except PlotSpecError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:                       # noqa: BLE001 - 任何绘图异常都要非 0 退出
        print(f"[FAIL] 绘图异常: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    print(f"run_id = {run_id}")
    for p in outs:
        if not p.is_file() or p.stat().st_size == 0:
            print(f"[FAIL] 图未生成或为空: {p}", file=sys.stderr)
            return 4
        print(f"已生成 {p}  ({p.stat().st_size / 1024:.1f} KB)")
    print(f"共 {len(outs)} 个文件（{len(langs)} 种语言 × 3 张图）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
