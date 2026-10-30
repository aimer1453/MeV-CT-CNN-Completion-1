"""P0-2 只读统计：零计数地板 5.50 与物理像素的混叠程度。

背景
----
`data/dataset.py` 的 `zero_log_fill`（当前全局 5.50）**只填充无效像素**
（`~(I>0 & E>0)`），有效像素保留物理 -log 值。因此若有效像素的 -log 值
≥ 5.50，这些像素就与填充值不可区分 —— 即"混叠"。

判据（用户给定）
----------------
- 有效像素 L 落在 [5.50, 6.20] 的占比 < 0.1% → 影响可忽略，保留全局 5.50
- > 1% → 5.50 与物理像素混叠，建议改 per-energy floor (6.30/5.60/4.90/4.50)

做法：逐档遍历全部 360 个投影，用 `log_ratio` 原式算 L，只统计有效像素。
"""
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
sys.path.insert(0, str(ROOT / "scripts"))

from data.dataset import log_ratio, valid_mask          # noqa: E402

FILL = 5.50
ENERGIES = ["1.5", "3", "4.5", "6"]
PROPOSED = {"1.5": 6.30, "3": 5.60, "4.5": 4.90, "6": 4.50}
ANGLES_CSV = resolve_csv("angles.csv")


def angle_rows(energy: str) -> list[dict]:
    with open(ANGLES_CSV, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["energy_MeV"] == energy]
    rows.sort(key=lambda r: float(r["angle_deg"]))
    if len(rows) != 360:
        raise SystemExit(f"[{energy}] angles.csv 期望 360 行，实得 {len(rows)}")
    return rows


print(f"zero_log_fill（当前全局）= {FILL}")
print(f"angles.csv = {ANGLES_CSV.name}（file 列已含 3_3/ 前缀，避开两套命名问题）")
print()
hdr = (f"{'E':>5} {'n_valid':>12} {'p99.9':>8} {'max':>8} "
       f"{'>=5.50':>9} {'[5.50,6.20]':>12} {'>=prop':>9} {'判定':>10}")
print(hdr)
print("-" * len(hdr))

summary = {}
for e in ENERGIES:
    rows = angle_rows(e)
    # empty_proj 路径：由 scan_dir 取（scan_dir 不含 3_3/ 前缀）
    scan_dirs = {r["scan_dir"] for r in rows}
    if len(scan_dirs) != 1:
        raise SystemExit(f"[{e}] 多个 scan_dir: {scan_dirs}")
    sd = scan_dirs.pop()
    empty_p = ROOT / "3_3" / sd / "results" / "empty_proj.npy"
    empty = np.load(empty_p).astype(np.float64)

    n_valid = 0
    n_ge_fill = 0
    n_in_band = 0
    n_ge_prop = 0
    # 分位数用蓄水池统计量不可靠，改为收集 row16 的全量 + 全图的汇总矩
    all_vals = []
    row16_vals = []
    for i, r in enumerate(rows):
        proj = np.load(ROOT / r["file"]).astype(np.float64)
        if proj.shape != (32, 1200):
            raise SystemExit(f"[{e}] 投影形状 {proj.shape}: {r['file']}")
        L = log_ratio(proj, empty).astype(np.float64)
        m = valid_mask(proj, empty.astype(np.float32))
        v = L[m]
        n_valid += v.size
        n_ge_fill += int((v >= FILL).sum())
        n_in_band += int(((v >= FILL) & (v <= 6.20)).sum())
        n_ge_prop += int((v >= PROPOSED[e]).sum())
        all_vals.append(v)
        row16_vals.append(L[16][m[16]])

    v = np.concatenate(all_vals)
    p999 = float(np.percentile(v, 99.9))
    vmax = float(v.max())
    f_ge = n_ge_fill / n_valid
    f_band = n_in_band / n_valid
    f_prop = n_ge_prop / n_valid
    verdict = "忽略" if f_band < 0.001 else ("改 per-energy" if f_band > 0.01 else "边界")
    summary[e] = {
        "n_valid": n_valid, "p999": p999, "max": vmax,
        "frac_ge_fill": f_ge, "frac_in_band_5.50_6.20": f_band,
        "frac_ge_proposed": f_prop, "proposed_floor": PROPOSED[e],
        "verdict": verdict,
    }
    print(f"{e:>5} {n_valid:>12,} {p999:>8.4f} {vmax:>8.4f} "
          f"{f_ge * 100:>8.4f}% {f_band * 100:>11.4f}% {f_prop * 100:>8.4f}% "
          f"{verdict:>10}")

print()
print("详细（含 row16 单行口径，即真正进入重建的那一行）:")
for e in ENERGIES:
    rows = angle_rows(e)
    sd = {r["scan_dir"] for r in rows}.pop()
    empty = np.load(ROOT / "3_3" / sd / "results" / "empty_proj.npy").astype(np.float64)
    vals = []
    for r in rows:
        proj = np.load(ROOT / r["file"]).astype(np.float64)
        L = log_ratio(proj, empty).astype(np.float64)
        m = valid_mask(proj, empty.astype(np.float32))[16]
        vals.append(L[16][m])
    v = np.concatenate(vals)
    n = v.size
    ge = int((v >= FILL).sum())
    band = int(((v >= FILL) & (v <= 6.20)).sum())
    print(f"  {e:>4} MeV row16: n={n:,} p99.9={np.percentile(v, 99.9):.4f} "
          f"max={v.max():.4f} | >=5.50: {ge} ({ge / n * 100:.4f}%) "
          f"[5.50,6.20]: {band} ({band / n * 100:.4f}%)")

print()
print("零计数（无效）像素占比，供对照:")
for e in ENERGIES:
    rows = angle_rows(e)
    sd = {r["scan_dir"] for r in rows}.pop()
    empty = np.load(ROOT / "3_3" / sd / "results" / "empty_proj.npy").astype(np.float64)
    n_tot = 0
    n_inv = 0
    for r in rows:
        proj = np.load(ROOT / r["file"]).astype(np.float64)
        m = valid_mask(proj, empty.astype(np.float32))
        n_tot += m.size
        n_inv += int((~m).sum())
    print(f"  {e:>4} MeV: 无效像素 {n_inv:,}/{n_tot:,} = {n_inv / n_tot * 100:.4f}%")
