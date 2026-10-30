#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
sanity_norm_stats.py  —— 只报告，不调整，不改任何文件
=====================================================
对当前 4 档（1.5/3.0/4.5/6.0 MeV）train split 的 1080 个投影，
统计非零像素在 -log 域的分位数与 mean/std，与旧训练集统计量
mean=0.370 / std=2.726 对比。

【为什么要给多种定义】
  投影里存的是连续强度（float32），不是整数计数：
  实测 1.5MeV 单文件 min=0 max=18.94，最小值 1.9e-5；6.0MeV max=43.68。
  "零计数像素"的判定（v==0）是明确的，但 **-log 用的是哪种底数、
  以及是否先乘回原始计数（×events_per_position=2e6），
  旧代码未迁移到位前无法从数据本身确定**。
  因此本脚本并列报告多组定义，不擅自选一个当成结论。

  其中：
    -ln(v)        = 直接对存储值取负自然对数
    -log10(v)     = -ln(v)/ln(10)
    -log2(v)      = -ln(v)/ln(2)
    -ln(v*2e6)    = -ln(v) - ln(2e6)，均值整体平移 ln(2e6)=14.51，std 不变
  后两者由 -ln 的统计量严格换算得到，不需要额外遍历。

【实现】
  无 numpy（当前环境未安装），用 array 解码 + 单遍累积 sum/sumsq、
  0.005 宽直方图求分位数（分位数误差 < 0.005，足以判断 |Δmean|>0.2）。

用法：
  python scripts/sanity_norm_stats.py
  python scripts/sanity_norm_stats.py --split train --max-files 200   # 快速试跑
"""

from __future__ import annotations

import argparse
import array
import csv
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
SPARSE = resolve_csv("sparse_targets.csv")

REF_MEAN, REF_STD = 0.370, 2.726
LN10 = math.log(10.0)
LN2 = math.log(2.0)
LN_2E6 = math.log(2e6)

N_PIX = 32 * 1200
BIN_W = 0.005
BIN_LO = -30.0
N_BIN = int(40.0 / BIN_W)


def load_npy(path: Path) -> array.array:
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(f"非 npy: {path}")
        major = f.read(2)[0]
        hlen = struct.unpack("<H", f.read(2))[0] if major == 1 else struct.unpack("<I", f.read(4))[0]
        f.read(hlen)
        a = array.array("f")
        a.fromfile(f, N_PIX)
    return a


class Acc:
    def __init__(self) -> None:
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0
        self.hist = [0] * N_BIN
        self.below = 0
        self.above = 0
        self.files = 0
        self.zero = 0

    def add(self, v: float) -> None:
        self.n += 1
        self.s += v
        self.s2 += v * v
        idx = int((v - BIN_LO) / BIN_W)
        if idx < 0:
            self.below += 1
        elif idx >= N_BIN:
            self.above += 1
        else:
            self.hist[idx] += 1

    def quant(self, q: float) -> float:
        """直方图分位数（线性插值到 bin 内），误差 < BIN_W"""
        target = q * self.n
        cum = 0
        for i, c in enumerate(self.hist):
            if cum + c >= target and c > 0:
                return BIN_LO + (i + (target - cum) / c) * BIN_W
            cum += c
        return float("nan")

    @property
    def mean(self) -> float:
        return self.s / self.n if self.n else float("nan")

    @property
    def std(self) -> float:
        if not self.n:
            return float("nan")
        m = self.mean
        var = self.s2 / self.n - m * m
        return math.sqrt(var) if var > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--max-files", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    by_e: dict[str, list[str]] = defaultdict(list)
    with open(SPARSE, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if args.split != "all" and r["split"] != args.split:
                continue
            by_e[r["energy_MeV"]].append(r["target_file"])

    energies = sorted(by_e, key=float)
    if args.max_files:
        for e in energies:
            by_e[e] = by_e[e][:args.max_files]

    print("=" * 104)
    print(f"样本：sparse_targets.csv 中 split={args.split} 的 target 文件")
    print(f"参考值：mean={REF_MEAN}  std={REF_STD}（旧训练集，排除零计数像素，-log 域）")
    print(f"零计数判定：v == 0（存储域）；-log 域直方图 bin 宽 {BIN_W}，分位数误差 < {BIN_W}")
    print("=" * 104)

    total = Acc()
    rows_out: list[tuple] = []

    for e in energies:
        acc = Acc()
        for rel in by_e[e]:
            arr = load_npy(ROOT / rel)
            acc.files += 1
            for v in arr:
                if v == 0.0:
                    acc.zero += 1
                    continue
                acc.add(-math.log(v))
        total.n += acc.n
        total.s += acc.s
        total.s2 += acc.s2
        total.zero += acc.zero
        total.files += acc.files
        for i, c in enumerate(acc.hist):
            total.hist[i] += c
        total.below += acc.below
        total.above += acc.above
        rows_out.append(("E", e, acc))

    rows_out.append(("ALL", "all", total))

    def line(tag: str, e: str, a: Acc) -> None:
        m, s = a.mean, a.std
        print(f"{tag:<5}{e:>5}{a.files:>7}{a.n:>11}{a.zero/a.n if a.n else 0:>9.1%}"
              f"{a.quant(0.01):>9.3f}{a.quant(0.50):>9.3f}{a.quant(0.99):>9.3f}"
              f"{m:>9.3f}{s:>9.3f}")

    print(f"{'档':<5}{'E':>5}{'files':>7}{'非零像素':>11}{'零占比':>9}"
          f"{'P1':>9}{'P50':>9}{'P99':>9}{'mean':>9}{'std':>9}   [单位: -ln(v)]")
    print("-" * 104)
    for tag, e, a in rows_out:
        line(tag, e, a)

    # ---------------- 多定义换算对比（基于 -ln 域统计量）
    print("\n" + "=" * 104)
    print("【多定义对比】与 0.370/2.726 的差值（全 4 档合并，train）")
    print(f"{'定义':<16}{'mean':>11}{'std':>10}{'Δmean':>11}{'Δstd%':>10}   判定")
    for name, shift, scale in (("-ln(v) 直接", 0.0, 1.0),
                               ("-log10(v)", 0.0, 1 / LN10),
                               ("-log2(v)", 0.0, 1 / LN2),
                               ("-ln(v*2e6)", -LN_2E6, 1.0)):
        m = total.mean * scale + shift
        s = total.std * scale
        dm = m - REF_MEAN
        ds = (s - REF_STD) / REF_STD
        verdict = "分布偏移待审" if (abs(dm) > 0.2 or abs(ds) > 0.20) else "接近"
        print(f"{name:<16}{m:>11.3f}{s:>10.3f}{dm:>+11.3f}{ds:>+10.1%}   {verdict}")

    # ---------------- 每档单独对比（-ln 直接口径）
    print("\n" + "=" * 104)
    print("【每档对比】-ln(v) 直接口径")
    print(f"{'E(MeV)':>8}{'mean':>10}{'std':>9}{'Δmean':>10}{'Δstd%':>10}   判定")
    for e, a in [(e, a) for tag, e, a in rows_out if tag == "E"]:
        m, s = a.mean, a.std
        dm, ds = m - REF_MEAN, (s - REF_STD) / REF_STD
        verdict = "分布偏移待审" if (abs(dm) > 0.2 or abs(ds) > 0.20) else "接近"
        print(f"{e:>8}{m:>10.3f}{s:>9.3f}{dm:>+10.3f}{ds:>+10.1%}   {verdict}")

    print("\n" + "=" * 104)
    print(f"直方图越界：低于 {BIN_LO} 的 {total.below} 个，高于 {BIN_LO+40} 的 {total.above} 个")
    print("注意：本脚本只报告，不修改任何统计量、不改 config、不改数据。")
    if total.n == 0:
        print("[ERROR] 没有统计到任何非零像素", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
