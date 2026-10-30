"""P0-2 补充：混叠像素按探测器行分解，定位 0.1089% 到底来自哪些行。"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/

from data.dataset import log_ratio, valid_mask          # noqa: E402

FILL = 5.50
ANGLES_CSV = resolve_csv("angles.csv")
PROPOSED = {"1.5": 6.30, "3": 5.60, "4.5": 4.90, "6": 4.50}


def angle_rows(energy):
    with open(ANGLES_CSV, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["energy_MeV"] == energy]
    rows.sort(key=lambda r: float(r["angle_deg"]))
    return rows


for e in ["1.5", "3"]:
    rows = angle_rows(e)
    sd = {r["scan_dir"] for r in rows}.pop()
    empty = np.load(ROOT / "3_3" / sd / "results" / "empty_proj.npy").astype(np.float64)

    per_row_valid = np.zeros(32, dtype=np.int64)
    per_row_band = np.zeros(32, dtype=np.int64)     # [5.50, 6.20]
    per_row_ge = np.zeros(32, dtype=np.int64)       # >= 5.50
    row_mean = np.zeros(32)
    for r in rows:
        proj = np.load(ROOT / r["file"]).astype(np.float64)
        L = log_ratio(proj, empty).astype(np.float64)
        m = valid_mask(proj, empty.astype(np.float32))
        row_mean += proj.mean(axis=1)
        for rr in range(32):
            v = L[rr][m[rr]]
            per_row_valid[rr] += v.size
            per_row_band[rr] += int(((v >= FILL) & (v <= 6.20)).sum())
            per_row_ge[rr] += int((v >= FILL).sum())
    row_mean /= len(rows)

    tot_v = per_row_valid.sum()
    tot_band = per_row_band.sum()
    tot_ge = per_row_ge.sum()
    print(f"=== {e} MeV（共 360 投影）===")
    print(f"有效像素总数 {tot_v:,}；[5.50,6.20] 共 {tot_band:,} "
          f"({tot_band / tot_v * 100:.4f}%)；>=5.50 共 {tot_ge:,} "
          f"({tot_ge / tot_v * 100:.4f}%)")
    print(f"{'row':>4} {'empty行均值':>12} {'有效px':>10} {'[5.50,6.20]':>12} {'占该行%':>9} "
          f"{'占全部band%':>12}")
    for rr in range(32):
        if per_row_band[rr] == 0 and per_row_ge[rr] == 0:
            continue
        print(f"{rr:>4} {row_mean[rr]:>12.4f} {per_row_valid[rr]:>10,} "
              f"{per_row_band[rr]:>12,} "
              f"{per_row_band[rr] / max(per_row_valid[rr], 1) * 100:>8.4f}% "
              f"{per_row_band[rr] / max(tot_band, 1) * 100:>11.2f}%")
    # 中心带（行 8..23）汇总
    c = slice(8, 24)
    cv = per_row_valid[c].sum()
    cb = per_row_band[c].sum()
    cg = per_row_ge[c].sum()
    print(f"中心带 行8..23: 有效 {cv:,}，[5.50,6.20] {cb:,} ({cb / cv * 100:.4f}%)，"
          f">=5.50 {cg:,} ({cg / cv * 100:.4f}%)")
    edge = np.r_[0:8, 24:32]
    ev = per_row_valid[edge].sum()
    eb = per_row_band[edge].sum()
    print(f"边缘带 行0..7+24..31: 有效 {ev:,}，[5.50,6.20] {eb:,} "
          f"({eb / ev * 100:.4f}%)，占全部 band 的 {eb / max(tot_band, 1) * 100:.2f}%")
    print(f"拟议 per-energy floor {PROPOSED[e]} 下的 >=floor 像素数：", end="")
    ge_p = 0
    for r in rows:
        proj = np.load(ROOT / r["file"]).astype(np.float64)
        L = log_ratio(proj, empty).astype(np.float64)
        m = valid_mask(proj, empty.astype(np.float32))
        ge_p += int((L[m] >= PROPOSED[e]).sum())
    print(f"{ge_p:,} ({ge_p / tot_v * 100:.4f}%)")
    print()
