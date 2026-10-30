#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/baseline_linear.py —— 投影域线性插值基线（方案 P）

方法
----
    target 投影 = (P[ch0] + P[ch_next]) / 2

为什么是「两点算术平均」：`sparse_targets.csv` 中 `target = ch0 + 1`、
`ch_next = ch0 + 2`（模 360），所以 target 恰好落在 ch0 与 ch_next 的**中点**，
线性插值退化为算术平均。不需要任何角度运算。

口径（与 CNN `train.py::evaluate` 逐项对齐）
--------------------------------------------
  - 同一个 `LOG_EPS = 1e-10`：`L = -log((I + eps) / (E + eps))`
  - 无效像素 `(I <= 0) | (E <= 0)` 在 **-log 域**填 `zero_log_fill`，之后不再归一化
    （基线不经过网络，物理域即 -log 域，故无需归一化/反归一化往返）
  - `mask = (I_target > 0) & (empty_proj > 0)`
  - `rmse_valid` 只在 mask 上算；`rmse_full` 在全 32x1200 像素上算
  - 分能量池化 `sqrt(sum(se) / sum(n))`，与 `EnergyAccumulator.summary` 同式
  - 同 test 集：`sparse_targets.csv` 中 `split == "test"`，4 档 × 45 = 180 样本

等价性说明：若改成「归一化域插值后再反归一化」，
`(((L0-m)/s) + ((L1-m)/s)) / 2 * s + m == (L0+L1)/2`，
与直接物理域插值数学等价，故直接算物理域。

输出（`runs/<cnn_run_id>_linear_interp/`）
------------------------------------------
  config.json          method="linear_interp"，optimizer/batch/lr/warmup = null
  train_log.csv        14 列，每档一行，epoch=-1，lr/loss_* = NaN
  test_metrics.json    结构与 CNN 完全一致
  test_per_sample.csv  11 列，与 CNN 完全一致，psnr/ssim = NaN

用法
----
  conda run -n ct_full python scripts/baseline_linear.py
  conda run -n ct_full python scripts/baseline_linear.py --cnn-run-id 20261022_2352_cnn_bs8_lr1e-4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/

from data.dataset import LOG_EPS, log_ratio, valid_mask  # noqa: E402

SPARSE_TARGETS = resolve_csv("sparse_targets.csv")
RUNS_DIR = ROOT / "runs"
DEFAULT_CNN_RUN_ID = "20261022_2352_cnn_bs8_lr1e-4"

ENERGIES: tuple[str, ...] = ("1.5", "3", "4.5", "6")
NAN = float("nan")

LOG_COLUMNS: tuple[str, ...] = (
    "epoch", "split", "energy_MeV", "n_samples", "coverage", "valid_coverage",
    "rmse_valid", "rmse_full", "psnr", "ssim", "n_invariant_warn", "lr",
    "loss_train", "loss_val",
)
PER_SAMPLE_COLUMNS: tuple[str, ...] = (
    "energy_MeV", "target_angle_deg", "split", "n_valid_pixels",
    "coverage", "valid_coverage", "rmse_valid", "rmse_full", "psnr", "ssim",
    "rmse_invariant_warn",
)

#: 与 CNN test_metrics.json 的 coverage/valid_coverage 交叉校验容差
COV_TOL = 1e-9
RMSE_TOL = 1e-9


class BaselineSpecError(RuntimeError):
    """输入或口径不满足时抛出（fail fast，不做兜底）。"""


# ------------------------------------------------------------------ 小工具
def energy_key(e: float) -> str:
    """1.5 -> '1.5'，3.0 -> '3'，4.5 -> '4.5'，6.0 -> '6'。与 train.py 同式。"""
    return f"{e:g}"


def git_info() -> tuple[str, bool]:
    def _run(*args: str) -> str:
        try:
            return subprocess.run(("git", *args), cwd=ROOT, capture_output=True,
                                  text=True, check=False).stdout.strip()
        except OSError:
            return ""

    commit = _run("rev-parse", "HEAD") or "UNKNOWN"
    dirty = bool(_run("status", "--porcelain"))
    return commit, dirty


def scan_dir_of(target_file: str) -> Path:
    """sparse_targets.csv 的 scan_dir 列是相对名（无 3_3/ 前缀），不能直接拼；
    必须由 target_file 取 parent.parent。"""
    return Path(target_file).parent.parent


def load_empty(scan_dir: Path, cache: dict[Path, np.ndarray]) -> np.ndarray:
    """empty_proj 在 {scan_dir}/results/empty_proj.npy，不在 scan 根目录。"""
    if scan_dir not in cache:
        p = scan_dir / "results" / "empty_proj.npy"
        if not p.is_file():
            raise BaselineSpecError(f"缺少 empty_proj: {p}")
        cache[scan_dir] = np.load(p)
    return cache[scan_dir]


def to_phys(rel_file: str, empty: np.ndarray, fill: float) -> np.ndarray:
    """单张投影 -> 物理域 -log 值（float64），无效像素填 fill。"""
    img = np.load(Path(rel_file)).astype(np.float32)
    L = log_ratio(img, empty).astype(np.float64)
    return np.where(np.isfinite(L), L, float(fill))


def load_test_rows() -> list[dict]:
    if not SPARSE_TARGETS.is_file():
        raise BaselineSpecError(f"缺少 {SPARSE_TARGETS}")
    with open(SPARSE_TARGETS, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "test"]
    if not rows:
        raise BaselineSpecError("sparse_targets.csv 中 split=test 没有样本")
    per = {e: sum(1 for r in rows if r["energy_MeV"] == e) for e in ENERGIES}
    if any(v == 0 for v in per.values()):
        raise BaselineSpecError(f"test 集分能量不完整: {per}")
    return rows


# ------------------------------------------------------------------ 主计算
def run_linear(cnn_run_id: str, fill: float) -> dict:
    rows = load_test_rows()
    empty_cache: dict[Path, np.ndarray] = {}

    per_energy: dict[str, dict[str, float]] = {
        e: {"n_samples": 0, "se_valid": 0.0, "n_valid": 0,
            "se_full": 0.0, "n_full": 0, "cov_sum": 0.0, "vcov_sum": 0.0}
        for e in ENERGIES
    }
    per_sample: list[dict] = []

    for r in rows:
        e = r["energy_MeV"]
        if e not in ENERGIES:
            raise BaselineSpecError(f"未知能量: {e!r}")
        scan = scan_dir_of(r["target_file"])
        empty = load_empty(scan, empty_cache)

        I_t = np.load(Path(r["target_file"])).astype(np.float32)
        I_0 = np.load(Path(r["ch0_file"])).astype(np.float32)
        I_n = np.load(Path(r["ch_next_file"])).astype(np.float32)

        L_t = to_phys(r["target_file"], empty, fill)
        L_0 = to_phys(r["ch0_file"], empty, fill)
        L_n = to_phys(r["ch_next_file"], empty, fill)

        # ---- 基线预测：target 恰在 ch0 与 ch_next 的中点
        pred = (L_0 + L_n) / 2.0

        mask = valid_mask(I_t, empty)
        sq = (pred - L_t) ** 2

        n_valid = int(mask.sum())
        n_full = int(sq.size)
        se_valid = float(sq[mask].sum())
        se_full = float(sq.sum())

        rmse_valid = math.sqrt(se_valid / n_valid) if n_valid else NAN
        rmse_full = math.sqrt(se_full / n_full) if n_full else NAN
        if not (math.isnan(rmse_valid) or math.isnan(rmse_full)):
            if rmse_full < rmse_valid - RMSE_TOL:
                raise BaselineSpecError(
                    f"rmse_full ({rmse_full:.9f}) < rmse_valid ({rmse_valid:.9f})，"
                    f"违反硬性关系: {r['target_file']}"
                )

        # coverage 两口径（数据属性，与模型无关）
        cov = float(np.count_nonzero(I_t > 0)) / float(I_t.size)
        vcov = float(np.count_nonzero(mask)) / float(mask.size)
        if cov < vcov - 1e-12:
            raise BaselineSpecError(f"coverage({cov}) < valid_coverage({vcov})")

        acc = per_energy[e]
        acc["n_samples"] += 1
        acc["se_valid"] += se_valid
        acc["n_valid"] += n_valid
        acc["se_full"] += se_full
        acc["n_full"] += n_full
        acc["cov_sum"] += cov
        acc["vcov_sum"] += vcov

        per_sample.append({
            "energy_MeV": e,
            "target_angle_deg": float(r["target_angle_deg"]),
            "split": "test",
            "n_valid_pixels": n_valid,
            "coverage": cov,
            "valid_coverage": vcov,
            "rmse_valid": rmse_valid,
            "rmse_full": rmse_full,
            "psnr": NAN,
            "ssim": NAN,
            "rmse_invariant_warn": False,
        })

    # ---- 分能量汇总（sqrt(sum(se)/sum(n))，与 EnergyAccumulator 同式）
    def _summary(acc: dict) -> dict:
        n = acc["n_samples"]
        rv = math.sqrt(acc["se_valid"] / acc["n_valid"]) if acc["n_valid"] else NAN
        rf = math.sqrt(acc["se_full"] / acc["n_full"]) if acc["n_full"] else NAN
        return {
            "n_samples": n,
            "coverage": acc["cov_sum"] / n if n else NAN,
            "valid_coverage": acc["vcov_sum"] / n if n else NAN,
            "rmse_valid": rv,
            "rmse_full": rf,
            "psnr": NAN,
            "ssim": NAN,
        }

    per_e = {e: _summary(per_energy[e]) for e in ENERGIES}

    # ---- overall 池化
    n_all = sum(v["n_samples"] for v in per_e.values())
    se_v = sum(v["rmse_valid"] ** 2 * per_energy[e]["n_valid"]
               for e, v in per_e.items() if not math.isnan(v["rmse_valid"]))
    n_v = sum(per_energy[e]["n_valid"] for e, v in per_e.items()
              if not math.isnan(v["rmse_valid"]))
    se_f = sum(v["rmse_full"] ** 2 * per_energy[e]["n_full"]
               for e, v in per_e.items() if not math.isnan(v["rmse_full"]))
    n_f = sum(per_energy[e]["n_full"] for e, v in per_e.items()
              if not math.isnan(v["rmse_full"]))
    overall = {
        "n_samples": n_all,
        "coverage": sum(v["coverage"] * v["n_samples"] for v in per_e.values()) / n_all,
        "valid_coverage": sum(v["valid_coverage"] * v["n_samples"] for v in per_e.values()) / n_all,
        "rmse_valid": math.sqrt(se_v / n_v) if n_v else NAN,
        "rmse_full": math.sqrt(se_f / n_f) if n_f else NAN,
        "psnr": NAN,
        "ssim": NAN,
    }
    if not (math.isnan(overall["rmse_valid"]) or math.isnan(overall["rmse_full"])):
        if overall["rmse_full"] < overall["rmse_valid"] - RMSE_TOL:
            raise BaselineSpecError(
                f"overall rmse_full ({overall['rmse_full']:.9f}) < "
                f"rmse_valid ({overall['rmse_valid']:.9f})，违反硬性关系"
            )

    return {"overall": overall, "per_energy": per_e, "per_sample": per_sample}


# ------------------------------------------------------------------ 交叉校验
def cross_check(metrics: dict, cnn_run_id: str) -> dict:
    """coverage / valid_coverage 是数据属性，基线与 CNN 必须逐档一致。"""
    p = RUNS_DIR / cnn_run_id / "test_metrics.json"
    if not p.is_file():
        return {"checked": False, "reason": f"缺少 {p}"}
    ref = json.loads(p.read_text(encoding="utf-8"))
    report: dict[str, dict[str, float]] = {}
    for e in ENERGIES:
        for k in ("coverage", "valid_coverage"):
            a = metrics["per_energy"][e][k]
            b = ref["per_energy"][e][k]
            d = abs(a - b)
            report[f"{e} MeV / {k}"] = {"baseline": a, "cnn": b, "abs_diff": d}
            if d > COV_TOL:
                raise BaselineSpecError(
                    f"[{e} MeV] {k} 与 CNN 不一致: baseline={a:.12f} cnn={b:.12f} "
                    f"diff={d:.3e} > {COV_TOL:g}；说明 test 集或口径不一致"
                )
    for k in ("coverage", "valid_coverage"):
        a = metrics["overall"][k]
        b = ref["overall"][k]
        report[f"overall / {k}"] = {"baseline": a, "cnn": b, "abs_diff": abs(a - b)}
        if abs(a - b) > COV_TOL:
            raise BaselineSpecError(
                f"overall {k} 与 CNN 不一致: baseline={a:.12f} cnn={b:.12f}"
            )
    return {"checked": True, "cnn_run_id": cnn_run_id, "tol": COV_TOL, "pairs": report}


# ------------------------------------------------------------------ 落盘
def write_outputs(out_dir: Path, cnn_run_id: str, fill: float, metrics: dict,
                  check: dict) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_info()
    cfg_cnn = json.loads((RUNS_DIR / cnn_run_id / "config.json").read_text(encoding="utf-8"))

    config = {
        "run_id": out_dir.name,
        "method": "linear_interp",
        "method_note": "target 投影 = (P[ch0] + P[ch_next]) / 2，两点算术平均；"
                       "target 恰为 ch0 与 ch_next 的中点，故线性插值退化为均值",
        "reference_run_id": cnn_run_id,
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": cfg_cnn.get("seed"),
        "batch": None,
        "lr": None,
        "optimizer": None,
        "warmup": None,
        "warmup_unit": None,
        "epochs": None,
        "null_fields_note": "基线无训练过程，optimizer/batch/lr/warmup/epochs 一律 null",
        "norm": cfg_cnn.get("norm"),
        "coverage_definition": cfg_cnn.get("coverage_definition"),
        "log_eps": LOG_EPS,
        "zero_log_fill": fill,
        "domain": "projection (-log physical)",
        "metric": {
            "primary": "rmse_valid",
            "also_report": ["rmse_full", "coverage", "valid_coverage"],
            "rmse_valid": "mask=(I_target>0)&(empty_proj>0) 上的 RMSE（物理域）",
            "rmse_full": "全部 32x1200 像素上的 RMSE（物理域）",
            "psnr_ssim": "投影域不适用，填 NaN",
        },
        "data": {"targets": "sparse_targets.csv", "split": "test",
                 "n_samples": metrics["overall"]["n_samples"],
                 "energies_MeV": list(ENERGIES)},
        "cross_check_with_cnn": {"checked": check["checked"]},
    }
    p_cfg = out_dir / "config.json"
    p_cfg.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- train_log.csv：14 列，每档一行，epoch=-1
    p_log = out_dir / "train_log.csv"
    with open(p_log, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        w.writeheader()
        for e in ENERGIES:
            v = metrics["per_energy"][e]
            w.writerow({
                "epoch": -1,
                "split": "test",
                "energy_MeV": e,
                "n_samples": v["n_samples"],
                "coverage": f"{v['coverage']:.6f}",
                "valid_coverage": f"{v['valid_coverage']:.6f}",
                "rmse_valid": f"{v['rmse_valid']:.6f}",
                "rmse_full": f"{v['rmse_full']:.6f}",
                "psnr": "NaN",
                "ssim": "NaN",
                "n_invariant_warn": 0,
                "lr": "NaN",
                "loss_train": "NaN",
                "loss_val": "NaN",
            })

    # ---- test_per_sample.csv：11 列
    p_ps = out_dir / "test_per_sample.csv"
    with open(p_ps, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PER_SAMPLE_COLUMNS)
        w.writeheader()
        for r in metrics["per_sample"]:
            out = {k: r[k] for k in PER_SAMPLE_COLUMNS}
            out["psnr"] = "NaN"
            out["ssim"] = "NaN"
            for k in ("coverage", "valid_coverage", "rmse_valid", "rmse_full"):
                out[k] = f"{out[k]:.6f}"
            out["target_angle_deg"] = f"{float(out['target_angle_deg']):g}"
            w.writerow(out)

    # ---- test_metrics.json：结构与 CNN 一致
    test_metrics = {
        "run_id": out_dir.name,
        "method": "linear_interp",
        "git_commit": commit,
        "git_dirty": dirty,
        "reference_run_id": cnn_run_id,
        "best_epoch": None,
        "best_val_rmse_valid_phys": None,
        "best_note": "基线无训练过程，不涉及 best 权重",
        "invariant_warn": {"n_warn": 0, "n_samples": metrics["overall"]["n_samples"],
                           "ratio": 0.0,
                           "note": "逐样本 rmse_full < rmse_valid 硬断言通过（本脚本直接 raise，不降级为告警）"},
        "overall": metrics["overall"],
        "per_energy": metrics["per_energy"],
        "psnr_ssim_note": "投影域基线不产出图像，PSNR/SSIM 填 NaN；图像域对比见 baseline_recon.py",
        "cross_check_with_cnn": check,
    }
    p_tm = out_dir / "test_metrics.json"
    p_tm.write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")

    return {"config": p_cfg, "train_log": p_log,
            "test_per_sample": p_ps, "test_metrics": p_tm}


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="投影域线性插值基线（方案 P）")
    ap.add_argument("--cnn-run-id", default=DEFAULT_CNN_RUN_ID,
                    help="参照的 CNN run（决定 norm 三值与交叉校验基准）")
    ap.add_argument("--suffix", default="linear_interp", help="输出目录后缀")
    args = ap.parse_args(argv)

    try:
        cfg_p = RUNS_DIR / args.cnn_run_id / "config.json"
        if not cfg_p.is_file():
            raise BaselineSpecError(f"缺少 {cfg_p}")
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        norm = cfg.get("norm") or {}
        for k in ("zero_log_fill",):
            if norm.get(k) is None:
                raise BaselineSpecError(f"config.json 的 norm.{k} 缺失，禁止使用默认值")
        fill = float(norm["zero_log_fill"])

        out_dir = RUNS_DIR / f"{args.cnn_run_id}_{args.suffix}"
        metrics = run_linear(args.cnn_run_id, fill)
        check = cross_check(metrics, args.cnn_run_id)
        paths = write_outputs(out_dir, args.cnn_run_id, fill, metrics, check)
    except BaselineSpecError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:                        # noqa: BLE001
        print(f"[FAIL] 异常: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    print("=" * 96)
    print(f"method = linear_interp   run_id = {out_dir.name}")
    print(f"zero_log_fill = {fill}   LOG_EPS = {LOG_EPS}   n = {metrics['overall']['n_samples']}")
    print("-" * 96)
    print(f"{'energy':>8} {'n':>4} {'coverage':>10} {'valid_cov':>10} "
          f"{'rmse_valid':>12} {'rmse_full':>12}")
    for e in ENERGIES:
        v = metrics["per_energy"][e]
        print(f"{e:>8} {v['n_samples']:>4} {v['coverage']:>10.6f} {v['valid_coverage']:>10.6f} "
              f"{v['rmse_valid']:>12.6f} {v['rmse_full']:>12.6f}")
    o = metrics["overall"]
    print(f"{'overall':>8} {o['n_samples']:>4} {o['coverage']:>10.6f} {o['valid_coverage']:>10.6f} "
          f"{o['rmse_valid']:>12.6f} {o['rmse_full']:>12.6f}")
    print("-" * 96)
    if check["checked"]:
        worst = max(v["abs_diff"] for v in check["pairs"].values())
        print(f"交叉校验：coverage / valid_coverage 与 CNN 逐档一致（最大偏差 {worst:.3e} <= {COV_TOL:g}）")
    else:
        print(f"交叉校验未执行：{check.get('reason')}")
    for k, p in paths.items():
        print(f"已写 {k:>16}: {p}  ({p.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
