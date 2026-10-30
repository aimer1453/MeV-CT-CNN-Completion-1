#!/usr/bin/env python
"""P0-1 配套：coverage vs 剂量因子 f（row 16 与全 32 行两条线）。

为什么单独算
------------
`compute_norm_stats.py` 只报**全 32 行**的 coverage/valid_coverage。但重建只用
第 16 行（`RECON_ROW=16`），而探测器在竖直方向极度不均（`empty_proj` 行均值
中心 22.49 / 边缘 0.36，约 60 倍），所以"全 32 行"的覆盖率塌陷并不代表成像行塌陷。
这一条正是"低剂量下 CNN 的优势从 row16 转移到全 32 行"的**入口解释**，必须单独给。

口径
----
  coverage       = (I > 0) 像素数 / 总像素数                  资产口径
  valid_coverage = (I > 0) & (E > 0) 像素数 / 总像素数        评估口径（= mask）
  两者都按 split=train 的 1080 个 target 的**逐样本均值**统计，与 config.json 的
  `coverage_reference` 完全同口径（f=1.0 复现 0.9437 / 0.9376 即为对齐校验）。
  row16 口径 = 只在 row 16 这一行上算，再对样本取均值。

用法
----
    python scripts/dose_coverage.py            # 四档全跑（f=1.0 用主表）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
RUNS = ROOT / "runs"
ROW = 16                      # = baseline_recon.RECON_ROW
NAN = float("nan")


def table_of(f: float) -> tuple[Path, Path]:
    """(csv, data_root)，f=1.0 用主表。"""
    if f == 1.0:
        return resolve_csv("sparse_targets.csv"), ROOT / "3_3"
    return ROOT / f"sparse_targets_dose_f{f:g}.csv", ROOT / f"3_3_dose_f{f:g}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.125])
    ap.add_argument("--splits", nargs="+", default=["train"],
                    help="统计哪些划分（默认 train，与 config 的 coverage_reference 同口径）")
    ap.add_argument("--out", default=str(RUNS / "_dose_coverage.json"))
    args = ap.parse_args()

    out: dict = {"row_px": ROW, "splits": args.splits,
                 "note": "逐样本均值；coverage=(I>0)，valid_coverage=(I>0)&(E>0)",
                 "factors": {}}

    for f in args.factors:
        csv_p, root = table_of(f)
        if not csv_p.is_file():
            raise SystemExit(f"[FATAL] 缺 {csv_p}")
        rows = [r for r in csv.DictReader(open(csv_p, newline="", encoding="utf-8"))
                if r["split"] in args.splits]
        if not rows:
            raise SystemExit(f"[FATAL] {csv_p.name} 在 split={args.splits} 下没有样本")
        per_e: dict[str, list] = {}
        emp_cache: dict[str, np.ndarray] = {}
        for r in rows:
            sd = r["scan_dir"]
            if sd not in emp_cache:
                emp_cache[sd] = np.load(root / sd / "results" / "empty_proj.npy").astype(np.float32)
            E = emp_cache[sd]
            I = np.load(ROOT / r["target_file"]).astype(np.float32)
            m = (I > 0) & (E > 0)
            c = (I > 0)
            v = per_e.setdefault(r["energy_MeV"], [])
            v.append((float(c.mean()), float(m.mean()),
                      float(c[ROW].mean()), float(m[ROW].mean()),
                      float((I <= 0).mean()), float((I[ROW] <= 0).mean())))
        agg = {}
        for e, v in per_e.items():
            a = np.array(v)
            agg[e] = {"n": len(v),
                      "coverage": float(a[:, 0].mean()),
                      "valid_coverage": float(a[:, 1].mean()),
                      "coverage_row16": float(a[:, 2].mean()),
                      "valid_coverage_row16": float(a[:, 3].mean()),
                      "zero_frac": float(a[:, 4].mean()),
                      "zero_frac_row16": float(a[:, 5].mean())}
        allv = np.array([x for v in per_e.values() for x in v])
        agg["ALL"] = {"n": len(allv),
                      "coverage": float(allv[:, 0].mean()),
                      "valid_coverage": float(allv[:, 1].mean()),
                      "coverage_row16": float(allv[:, 2].mean()),
                      "valid_coverage_row16": float(allv[:, 3].mean()),
                      "zero_frac": float(allv[:, 4].mean()),
                      "zero_frac_row16": float(allv[:, 5].mean())}
        out["factors"][f"{f:g}"] = {"csv": csv_p.name, "root": root.name,
                                    "per_energy": agg, "_all": agg["ALL"]}
        a = agg["ALL"]
        print(f"f={f:<6g} n={a['n']:4d}  "
              f"all32 cov={a['coverage']*100:5.2f}%  vcov={a['valid_coverage']*100:5.2f}%   "
              f"row16 cov={a['coverage_row16']*100:6.2f}%  vcov={a['valid_coverage_row16']*100:6.2f}%")

    # ---- 校验：f=1.0 的 all32 valid_coverage 应与 config.json 的 0.9376 一致
    ref = 0.9376
    got = out["factors"]["1" if "1" in out["factors"] else "1.0"]["_all"]["valid_coverage"]
    print(f"\n[校验] f=1.0 all32 valid_coverage = {got:.4f}（config.json 参考 {ref}）"
          f"  {'OK' if abs(got - ref) < 5e-4 else '差异偏大，检查口径'}")

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[json] {Path(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
