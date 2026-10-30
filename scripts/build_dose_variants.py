#!/usr/bin/env python
"""gen 泊松降采样数据镜像（P0-1 用，不重跑 Geant4）。

模型
----
把每个投影文件的每个像素当作**期望计数** N，按剂量因子 f 抽稀：

    I_f = Poisson(f · N) / f        （E[N] = N，Var = N / f）

除以 f 保证无偏：E[I_f] = N，所以 -log(I_f/E) 仍是 L 的无偏估计，
只是方差放大 1/f。等价于曝光时间乘以 f。

**空场 E 不抽稀**：`empty_proj.npy` 视为高剂量下校准好的 air scan（文献里低剂量
CT 仿真的标准假设）。理由有二：
  ① 若 E 也抽稀，它的噪声是**逐 scan_dir 固定**的，360 个角度共用同一个实现，
     不会在视图间平均掉，会直接变成环形/结构性伪影，混淆"低剂量"本身的效应；
  ② 只抽稀 I 时，-log 的方差 ≈ 1/(f·N)，与"曝光时间减半"的物理一致。
本文件的配置记录在 config 的 `dose` 段里，可复核。

输出
----
    3_3_dose_f<f>/<scan_dir>/projections/*.npy    抽稀后的投影
    3_3_dose_f<f>/<scan_dir>/results/empty_proj.npy  原样拷贝（数据管线要求）
    sparse_targets_dose_f<f>.csv                  路径改写到新根的数据表

用法
----
    python scripts/build_dose_variants.py --factors 0.5 0.25 0.125
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
SRC_CSV = resolve_csv("sparse_targets.csv")
FILE_COLS = ("target_file", "ch0_file", "ch_prev_file", "ch_next_file")
SRC_ROOT = "3_3/"


def root_name(f: float) -> str:
    return f"3_3_dose_f{f:g}"


def rng_for(f: float, rel: str) -> np.random.Generator:
    """逐文件确定性的 RNG：同一文件在任何 split / 任何通道里都得到同一个抽稀实现。"""
    h = hashlib.sha256(f"{f:.6f}|{rel}".encode()).hexdigest()[:16]
    return np.random.default_rng(int(h, 16))


def thin(arr: np.ndarray, f: float, rng: np.random.Generator) -> np.ndarray:
    lam = np.clip(arr.astype(np.float64), 0.0, None) * f
    return (rng.poisson(lam) / f).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="泊松降采样数据镜像")
    ap.add_argument("--factors", type=float, nargs="+", default=[0.5, 0.25, 0.125])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(SRC_CSV, newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"[FATAL] 空表 {SRC_CSV}")
    files = sorted({r[c] for r in rows for c in FILE_COLS})
    scan_dirs = sorted({r["scan_dir"] for r in rows})
    print(f"源表 {SRC_CSV.name}: {len(rows)} 行, {len(files)} 个唯一投影文件, "
          f"{len(scan_dirs)} 个 scan_dir")
    for s in scan_dirs:
        print(f"    {s}")

    for f in args.factors:
        if not (0.0 < f < 1.0):
            raise SystemExit(f"[FATAL] f 必须在 (0,1) 内（1.0 用现有 canonical run）: {f}")
        rn = root_name(f)
        dst_root = ROOT / rn
        t0 = time.time()
        n_zero_src = n_zero_dst = n_pix = 0
        print(f"\n=== f = {f:g} -> {rn}/ ===")
        for i, rel in enumerate(files):
            src = ROOT / rel
            if not src.is_file():
                raise SystemExit(f"[FATAL] 缺源文件 {src}")
            arr = np.load(src)
            if arr.shape != (32, 1200):
                raise SystemExit(f"[FATAL] shape {arr.shape} != (32,1200): {src}")
            out = dst_root / rel[len(SRC_ROOT):] if rel.startswith(SRC_ROOT) else dst_root / rel
            n_zero_src += int(np.count_nonzero(arr <= 0))
            n_pix += arr.size
            if not args.dry_run:
                out.parent.mkdir(parents=True, exist_ok=True)
                b = thin(arr, f, rng_for(f, rel))
                n_zero_dst += int(np.count_nonzero(b <= 0))
                np.save(out, b)
            if (i + 1) % 360 == 0:
                print(f"    {i + 1}/{len(files)} ...", flush=True)

        for s in scan_dirs:
            src_e = ROOT / "3_3" / s / "results" / "empty_proj.npy"
            dst_e = dst_root / s / "results" / "empty_proj.npy"
            if not src_e.is_file():
                raise SystemExit(f"[FATAL] 缺空场 {src_e}")
            if not args.dry_run:
                dst_e.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_e, dst_e)

        out_csv = resolve_csv(f"sparse_targets_dose_f{f:g}.csv", for_write=True)
        if not args.dry_run:
            w = list(rows[0].keys())
            with open(out_csv, "w", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(fh, fieldnames=w)
                wr.writeheader()
                for r in rows:
                    r2 = dict(r)
                    for c in FILE_COLS:
                        r2[c] = r[c].replace(SRC_ROOT, rn + "/", 1)
                    wr.writerow(r2)

        zs = 100.0 * n_zero_src / max(1, n_pix)
        zd = 100.0 * n_zero_dst / max(1, n_pix) if n_pix else float("nan")
        print(f"    零计数像素占比: 源 {zs:.2f} % -> f={f:g} {zd:.2f} %"
              f"   （{time.time() - t0:.1f} s）")
        print(f"    CSV: {out_csv.name}")
    if args.dry_run:
        print("\n[dry-run] 未写任何文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
