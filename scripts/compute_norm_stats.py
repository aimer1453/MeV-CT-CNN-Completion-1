#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/compute_norm_stats.py —— 计算 norm 统计量与 ZERO_LOG_FILL（只报告，不写 config）
=====================================================================================
口径（用户裁决，不得改）：
  数据范围：sparse_targets.csv 中 split=train 的 1080 个 target 文件
  计算域  ：L = -log((I + 1e-10) / (E + 1e-10))，E = 同能量目录的 empty_proj
  无效像素：(I <= 0) | (E <= 0)

两套统计：
  valid  ：只用有效像素的值            —— 用于训练归一化的那一套
  all    ：全部像素；无效像素按 ZERO_LOG_FILL 填充后的值
           （即网络真正看到的域。因 fill >= p99.9 但可能 < max(valid)，
             故用「排序合并」精确求分位数，不生成长数组）

ZERO_LOG_FILL = ceil_to_0.1(valid 的 p99.9)，分档 + 全局都算
zero_fill_normalized = (ZERO_LOG_FILL - norm_mean) / norm_std

额外对比：split=train 的输入文件并集（ch_prev_file / ch0_file / ch_next_file 去重）
的同口径 valid mean/std 与 1080 个 target 对比；
若 |Δmean| > 0.05 或 |Δstd|/std > 5% 则标记 BLOCK，等裁决。

用法：
  conda run -n ct_full python scripts/compute_norm_stats.py
输出：
  终端完整报告 + docs/norm_stats_proposal.json（提案，不是 config）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
SPARSE = resolve_csv("sparse_targets.csv")
OUT_JSON = ROOT / "docs" / "norm_stats_proposal.json"

LOG_EPS = 1e-10
IMAGE_SHAPE = (32, 1200)
HIST_LO, HIST_HI, HIST_BW = 0.0, 8.0, 0.05
DELTA_MEAN_TH, DELTA_STD_REL_TH = 0.05, 0.05
ENERGIES = ("1.5", "3", "4.5", "6")


def ceil_to_0_1(x: float) -> float:
    """向上取一位小数；先 round 掉浮点噪声，避免 4.7 被推成 4.8。"""
    return math.ceil(round(x, 6) * 10.0) / 10.0


def log_domain(img: np.ndarray, empty: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = img.astype(np.float64)
    b = empty.astype(np.float64)
    L = -np.log((a + LOG_EPS) / (b + LOG_EPS))
    valid = (a > 0.0) & (b > 0.0)
    return L, valid


def pct_merged(sorted_valid: np.ndarray, n_invalid: int, fill: float, q: float) -> float:
    """全像素分位数：把 n_invalid 个常数 fill 与 sorted_valid 合并后的精确分位数。"""
    nv = sorted_valid.size
    N = nv + n_invalid
    if N == 0:
        return float("nan")
    k = int(np.searchsorted(sorted_valid, fill, side="left"))  # 严格小于 fill 的有效值个数

    def at(p: int) -> float:
        if p < k:
            return float(sorted_valid[p])
        if p < k + n_invalid:
            return float(fill)
        return float(sorted_valid[p - n_invalid])

    pos = q * (N - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, N - 1)
    a, b = at(lo), at(hi)
    return a + (b - a) * (pos - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_JSON))
    # 泊松降采样实验（P0-1）：--csv 指定降采样后的数据表；默认行为不变
    ap.add_argument("--csv", default=None,
                    help="数据表路径（默认 sparse_targets.csv）。"
                         "P0-1 剂量实验用 sparse_targets_dose_f<f>.csv")
    args = ap.parse_args()
    sparse = Path(args.csv) if args.csv else SPARSE
    if not sparse.is_file():
        raise SystemExit(f"[FATAL] 缺数据表 {sparse}")
    print(f"[CSV] {sparse.name}")

    rows_train: list[dict] = []
    rows_all: list[dict] = []
    with open(sparse, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows_all.append(r)
            if r["split"] == "train":
                rows_train.append(r)
    if len(rows_train) != 1080:
        print(f"[FATAL] split=train 样本数 {len(rows_train)} != 1080", file=sys.stderr)
        return 2

    # scan_dir -> empty_proj（按能量）
    scan_path: dict[str, Path] = {}
    for r in rows_train:
        scan_path.setdefault(r["scan_dir"], (ROOT / r["target_file"]).parent.parent)

    empty_cache: dict[str, np.ndarray] = {}

    def empty_of(scan_dir: str) -> np.ndarray:
        if scan_dir not in empty_cache:
            p = scan_path[scan_dir] / "results" / "empty_proj.npy"
            empty_cache[scan_dir] = np.load(p).astype(np.float64)
        return empty_cache[scan_dir]

    # ---------------------------------------------------------------- 主统计
    bins = np.arange(HIST_LO, HIST_HI + HIST_BW * 0.5, HIST_BW)
    per_e_valid: dict[str, list[np.ndarray]] = defaultdict(list)
    per_e_ninvalid: dict[str, int] = defaultdict(int)
    per_e_ncov: dict[str, int] = defaultdict(int)   # I_target > 0 计数（资产口径）
    per_e_hist: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(bins) - 1, dtype=np.int64))
    per_e_out: dict[str, dict] = defaultdict(lambda: {"below": 0, "above": 0})

    for i, r in enumerate(rows_train):
        e = r["energy_MeV"]
        emp = empty_of(r["scan_dir"])
        img = np.load(ROOT / r["target_file"]).astype(np.float32)
        if img.shape != IMAGE_SHAPE:
            print(f"[FATAL] shape {img.shape}: {r['target_file']}", file=sys.stderr)
            return 2
        L, valid = log_domain(img, emp)
        vals = L[valid]
        per_e_valid[e].append(vals)
        per_e_ninvalid[e] += int((~valid).sum())
        per_e_ncov[e] += int(np.count_nonzero(img > 0))
        h, _ = np.histogram(vals, bins=bins)
        per_e_hist[e] += h
        per_e_out[e]["below"] += int((vals < HIST_LO).sum())
        per_e_out[e]["above"] += int((vals > HIST_HI).sum())
        if (i + 1) % 270 == 0:
            print(f"  ... {i+1}/1080 ({e} MeV 完成)", file=sys.stderr)

    # ---------------------------------------------------------------- 汇总计算
    def summarize(valid_arr: np.ndarray, n_invalid: int, n_cov: int, fill: float) -> dict:
        vs = np.sort(valid_arr)
        nv, N = int(vs.size), int(vs.size + n_invalid)
        s, s2 = float(vs.astype(np.float64).sum()), float((vs.astype(np.float64) ** 2).sum())
        vm, vstd = s / nv, math.sqrt(max(s2 / nv - (s / nv) ** 2, 0.0))
        am = (s + n_invalid * fill) / N
        astd = math.sqrt(max((s2 + n_invalid * fill * fill) / N - am * am, 0.0))
        return {
            "n_valid": nv, "n_invalid": n_invalid, "n_total": N,
            "coverage": n_cov / N,          # 资产口径 I_target > 0 / 总像素
            "valid_coverage": nv / N,       # 评估口径 (I>0)&(E>0) / 总像素
            "valid": {"p1": float(np.percentile(vs, 1)), "p50": float(np.percentile(vs, 50)),
                      "p99": float(np.percentile(vs, 99)), "p99.9": float(np.percentile(vs, 99.9)),
                      "mean": vm, "std": vstd},
            "all": {"p1": pct_merged(vs, n_invalid, fill, 0.01),
                    "p50": pct_merged(vs, n_invalid, fill, 0.50),
                    "p99": pct_merged(vs, n_invalid, fill, 0.99),
                    "mean": am, "std": astd},
        }

    per_energy: dict[str, dict] = {}
    all_valid_list: list[np.ndarray] = []
    total_invalid = 0
    total_cov = 0
    for e in ENERGIES:
        arr = np.concatenate(per_e_valid[e])
        all_valid_list.append(arr)
        total_invalid += per_e_ninvalid[e]
        total_cov += per_e_ncov[e]
        p999 = float(np.percentile(arr, 99.9))
        fill = ceil_to_0_1(p999)
        d = summarize(arr, per_e_ninvalid[e], per_e_ncov[e], fill)
        d["p99.9_raw"] = p999
        d["zero_log_fill"] = fill
        d["hist"] = {
            "bin_edges": bins.tolist(),
            "counts": per_e_hist[e].tolist(),
            "below": per_e_out[e]["below"],
            "above": per_e_out[e]["above"],
        }
        per_energy[e] = d

    glob_valid = np.concatenate(all_valid_list)
    glob_p999 = float(np.percentile(glob_valid, 99.9))
    glob_fill = ceil_to_0_1(glob_p999)
    glob = summarize(glob_valid, total_invalid, total_cov, glob_fill)
    glob["p99.9_raw"] = glob_p999
    glob["zero_log_fill"] = glob_fill

    # ---------------------------------------------------------------- 填充后归一化值
    for e in ENERGIES:
        per_energy[e]["zero_fill_normalized"] = (
            (per_energy[e]["zero_log_fill"] - per_energy[e]["valid"]["mean"])
            / per_energy[e]["valid"]["std"]
        )
    glob["zero_fill_normalized"] = ((glob_fill - glob["valid"]["mean"]) / glob["valid"]["std"])

    # ---------------------------------------------------------------- 输入文件并集
    union: dict[str, str] = {}
    for r in rows_train:
        for k in ("ch_prev_file", "ch0_file", "ch_next_file"):
            union.setdefault(r[k], r["scan_dir"])
    u_sum: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])  # sum, sumsq, n
    for rel, scan_dir in union.items():
        emp = empty_of(scan_dir)
        img = np.load(ROOT / rel).astype(np.float32)
        L, valid = log_domain(img, emp)
        v = L[valid].astype(np.float64)
        e = next(r["energy_MeV"] for r in rows_train if r["scan_dir"] == scan_dir)
        acc = u_sum[e]
        acc[0] += float(v.sum())
        acc[1] += float((v ** 2).sum())
        acc[2] += int(v.size)

    input_union: dict[str, dict] = {}
    for e in ENERGIES:
        s, s2, n = u_sum[e]
        m = s / n
        input_union[e] = {"n_files": None, "n_valid": n, "mean": m,
                          "std": math.sqrt(max(s2 / n - m * m, 0.0))}
    ts, ts2, tn = (sum(u_sum[e][0] for e in ENERGIES), sum(u_sum[e][1] for e in ENERGIES),
                   sum(u_sum[e][2] for e in ENERGIES))
    tm = ts / tn
    input_union["ALL"] = {"n_files": len(union), "n_valid": tn, "mean": tm,
                          "std": math.sqrt(max(ts2 / tn - tm * tm, 0.0))}
    for e in ENERGIES:
        input_union[e]["n_files"] = sum(1 for sd in union.values()
                                        if sd.startswith(f"energy_scan_{e}MeV"))

    # ---------------------------------------------------------------- 对比
    comparison: dict[str, dict] = {}
    blocked = False
    for e in list(ENERGIES) + ["ALL"]:
        tgt = per_energy[e]["valid"] if e != "ALL" else glob["valid"]
        uni = input_union[e]
        dm = uni["mean"] - tgt["mean"]
        ds = (uni["std"] - tgt["std"]) / tgt["std"]
        bad = abs(dm) > DELTA_MEAN_TH or abs(ds) > DELTA_STD_REL_TH
        blocked = blocked or bad
        comparison[e] = {"target_mean": tgt["mean"], "target_std": tgt["std"],
                         "input_mean": uni["mean"], "input_std": uni["std"],
                         "delta_mean": dm, "delta_std_rel": ds, "exceeds": bad}

    # ---------------------------------------------------------------- 报告
    W = 108
    print("=" * W)
    print(f"norm 统计（{sparse.name} split=train，len(rows_train)=%d 个 target 文件）" % len(rows_train))
    print(f"域: -log((I+{LOG_EPS:g})/(E+{LOG_EPS:g}))   无效像素: (I<=0)|(E<=0)")
    print("=" * W)
    print(f"{'':<6}{'n_valid':>10}{'n_invalid':>10}{'':>3}"
          f"{'-- valid（训练用） --':>34}{'-- all（含填充） --':>34}")
    print(f"{'E':<6}{'':>10}{'':>10}{'':>3}{'P1':>8}{'P50':>8}{'P99':>8}{'mean':>9}"
          f"{'P1':>8}{'P50':>8}{'P99':>8}{'mean':>9}")
    print("-" * W)
    for e in list(ENERGIES) + ["ALL"]:
        d = per_energy[e] if e != "ALL" else glob
        v, a = d["valid"], d["all"]
        print(f"{e:<6}{d['n_valid']:>10}{d['n_invalid']:>10}{'':>3}"
              f"{v['p1']:>8.3f}{v['p50']:>8.3f}{v['p99']:>8.3f}{v['mean']:>9.4f}"
              f"{a['p1']:>8.3f}{a['p50']:>8.3f}{a['p99']:>8.3f}{a['mean']:>9.4f}")
    print("-" * W)
    print(f"{'E':<6}{'valid std':>12}{'all std':>10}{'p99.9(valid)':>14}"
          f"{'ZERO_LOG_FILL':>15}{'zero_fill_norm':>16}")
    for e in list(ENERGIES) + ["ALL"]:
        d = per_energy[e] if e != "ALL" else glob
        print(f"{e:<6}{d['valid']['std']:>12.4f}{d['all']['std']:>10.4f}"
              f"{d['p99.9_raw']:>14.4f}{d['zero_log_fill']:>15.2f}"
              f"{d['zero_fill_normalized']:>+16.4f}")

    print("\n" + "=" * W)
    print("coverage 两口径（全 1080 样本，逐样本等权）")
    print(f"{'E':<6}{'coverage':>12}{'valid_coverage':>16}{'差值':>10}")
    for e in list(ENERGIES) + ["ALL"]:
        d = per_energy[e] if e != "ALL" else glob
        print(f"{e:<6}{d['coverage']:>12.4f}{d['valid_coverage']:>16.4f}"
              f"{d['coverage'] - d['valid_coverage']:>10.4f}")

    print("\n" + "=" * W)
    print("输入文件并集 vs 1080 target（valid 口径 mean/std）")
    print(f"  并集去重后文件数 = {len(union)}（3 通道 × 1080 行去重）")
    print(f"{'E':<6}{'target mean':>13}{'input mean':>12}{'Δmean':>10}{'':>2}"
          f"{'target std':>12}{'input std':>11}{'Δstd%':>9}   判定")
    for e in list(ENERGIES) + ["ALL"]:
        c = comparison[e]
        flag = "BLOCK 超阈" if c["exceeds"] else "OK"
        print(f"{e:<6}{c['target_mean']:>13.4f}{c['input_mean']:>12.4f}"
              f"{c['delta_mean']:>+10.4f}{'':>2}{c['target_std']:>12.4f}"
              f"{c['input_std']:>11.4f}{c['delta_std_rel']:>+9.2%}   {flag}")
    print(f"阈值：|Δmean| > {DELTA_MEAN_TH} 或 |Δstd|/std > {DELTA_STD_REL_TH:.0%}")
    print("=" * W)
    print(f"[{'BLOCK' if blocked else 'OK'}] 输入并集对比："
          f"{'存在超阈项，需裁决后才能定 norm 来源' if blocked else '全部在阈值内'}")

    payload = {
        "meta": {
            "source": f"{sparse.name} split=train",
            "n_files": len(rows_train),
            "domain": f"-log((I+{LOG_EPS:g})/(E+{LOG_EPS:g}))",
            "invalid": "(I<=0)|(E<=0)",
            "all_pixel_definition": "全部像素；无效像素按 ZERO_LOG_FILL 填充后统计（网络所见域）",
            "zero_log_fill_rule": "ceil_to_0.1(valid p99.9)",
            "thresholds": {"delta_mean": DELTA_MEAN_TH, "delta_std_rel": DELTA_STD_REL_TH},
            "energy_filter": list(ENERGIES),
            "git_note": "提案文件，非 config；需裁决后才可写入 config.json",
        },
        "per_energy": per_energy,
        "global": glob,
        "input_union": input_union,
        "comparison": comparison,
        "blocked": blocked,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出提案 JSON: {out}")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
