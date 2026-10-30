#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_data_manifest.py
=======================
生成 data_manifest.csv：数据资产清单（只跟踪清单，不跟踪数据本身）。

【唯一真相来源规则 — 必须遵守】
  1. energy_MeV 只从目录名 energy_scan_<E>MeV_... 解析。
     !!! 严禁读取 scan_parameters.json 的 energy_MeV 字段 !!!
     实测该字段在全部 16 个目录中均硬编码为 3.0（残留废弃字段），
     真实能量只体现在 macros/*.mac 的 /gps/ene/mono 和目录名里。
  2. angle_deg 只从投影文件名解析。
     !!! 严禁读取 scan_parameters.json 的 angles 字段 !!!
     非均匀角度扫描下该字段不可靠。

【两套命名兼容】
  A) proj_a000_z+0.npy      -> angle=0.0    （9 个目录：0.5/1.0/1.5/2.0/2.5/3.0/4.0/5.0/6.0 MeV）
  B) proj_a0000.0_z+0.npy   -> angle=0.0    （7 个目录：4.5/5.5/6.5/7.0/8.0/9.0/10.0 MeV，_z0.0 系列）
  两者角度均为 0.0，正则 proj_a([0-9.]+)_z 同时兼容。

用法：
  python scripts/build_data_manifest.py            # 全量 16 档
  python scripts/build_data_manifest.py --first-run-only   # 只输出 4 档
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------- 常量

DATA_ROOT = Path(__file__).resolve().parent.parent / "3_3"
OUT_CSV = Path(__file__).resolve().parent.parent / "data_manifest.csv"

# 第一段 5 epoch 只用的 4 个能量档（用户裁决，见项目决策记录）
FIRST_RUN_ENERGIES = (1.5, 3.0, 4.5, 6.0)

# 目录名 -> 能量
RE_DIR_ENERGY = re.compile(r"^energy_scan_([0-9.]+)MeV_")

# 文件名 -> 角度（兼容 a000 与 a0000.0 两套）
RE_FILE_ANGLE = re.compile(r"^proj_a([0-9.]+)_z[+-]?([0-9.]+)\.npy$")

EXPECTED_BYTES = 32 * 1200 * 4 + 128  # (32,1200) float32 + npy header


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def scan_dirs(data_root: Path) -> list[Path]:
    dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir() and d.name.startswith("energy_scan_")]
    return dirs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-run-only", action="store_true",
                    help="只输出 1.5/3.0/4.5/6.0 MeV 这 4 档（1440 行）")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_csv = Path(args.out)
    if not data_root.is_dir():
        print(f"[FATAL] 数据根不存在: {data_root}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    per_dir_stat: list[tuple] = []
    errors: list[str] = []
    naming_hits = {"A(a000)": 0, "B(a0000.0)": 0}

    for d in scan_dirs(data_root):
        m = RE_DIR_ENERGY.match(d.name)
        if not m:
            errors.append(f"目录名无法解析能量: {d.name}")
            continue
        energy = float(m.group(1))
        if args.first_run_only and energy not in FIRST_RUN_ENERGIES:
            continue

        proj_dir = d / "projections"
        if not proj_dir.is_dir():
            errors.append(f"缺少 projections/: {d.name}")
            continue

        files = sorted(proj_dir.glob("*.npy"))
        angles: list[float] = []
        bad: list[str] = []
        local_a = local_b = 0

        for f in files:
            fm = RE_FILE_ANGLE.match(f.name)
            if not fm:
                bad.append(f.name)
                continue
            angle = float(fm.group(1))
            # 两套命名计数
            if "." in fm.group(1):
                local_b += 1
            else:
                local_a += 1
            size = f.stat().st_size
            if size != EXPECTED_BYTES:
                errors.append(f"文件大小异常 {size} != {EXPECTED_BYTES}: {f}")
            rows.append({
                "file": f.relative_to(data_root.parent).as_posix(),
                "scan_dir": d.name,
                "energy_MeV": f"{energy:g}",
                "angle_deg": f"{angle:g}",
                "size_bytes": size,
                "md5": md5_of(f),
                "in_first_run": 1 if energy in FIRST_RUN_ENERGIES else 0,
            })
            angles.append(angle)

        naming_hits["A(a000)"] += local_a
        naming_hits["B(a0000.0)"] += local_b
        per_dir_stat.append((d.name, energy, len(files), len(angles), local_a, local_b, bad))

    # ---------------- 写 CSV
    fields = ["file", "scan_dir", "energy_MeV", "angle_deg", "size_bytes", "md5", "in_first_run"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---------------- 报告
    print("=" * 108)
    print(f"{'scan_dir':<56}{'E(MeV)':>8}{'files':>7}{'parsed':>8}{'A':>5}{'B':>5}  状态")
    print("=" * 108)
    for name, energy, nf, na, la, lb, bad in per_dir_stat:
        st = "OK" if (nf == na and not bad) else f"BAD(未解析{len(bad)})"
        print(f"{name:<56}{energy:>8g}{nf:>7}{na:>8}{la:>5}{lb:>5}  {st}")
    print("=" * 108)
    print(f"命名命中: A(proj_a000_z+0)={naming_hits['A(a000)']}  B(proj_a0000.0_z+0)={naming_hits['B(a0000.0)']}")
    print(f"目录数={len(per_dir_stat)}  文件数={len(rows)}  首轮(4档)行数={sum(r['in_first_run'] for r in rows)}")
    total_bytes = sum(r["size_bytes"] for r in rows)
    print(f"总大小={total_bytes} B ({total_bytes/1024/1024:.1f} MiB)")
    print(f"输出: {out_csv}")

    if errors:
        print("\n[ERRORS]")
        for e in errors[:30]:
            print("  ! " + e)
        if len(errors) > 30:
            print(f"  ... 共 {len(errors)} 条")
        return 1
    print("\n[OK] 无异常")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
