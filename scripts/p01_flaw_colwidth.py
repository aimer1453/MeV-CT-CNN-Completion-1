"""P0-1 收尾：中心缺陷的列向宽度（360 角累加 row16 列比值）。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"


def main():
    sf = np.zeros(1200)
    su = np.zeros(1200)
    for a in range(360):
        pf = np.load(FLAW / "projections" / f"proj_a{a:06.1f}_z+0.npy").astype(np.float64)[16]
        pu = np.load(UNIF / "projections" / f"proj_a{a:03d}_z+0.npy").astype(np.float64)[16]
        sf += pf
        su += pu
    r = sf / su
    print("col 比值 >1.2 的连续段（360 角累加 row16）:")
    idx = np.where(r > 1.2)[0]
    if len(idx):
        segs = []
        s = idx[0]
        for k in range(1, len(idx)):
            if idx[k] != idx[k - 1] + 1:
                segs.append((int(s), int(idx[k - 1])))
                s = idx[k]
        segs.append((int(s), int(idx[-1])))
        for a, b in segs:
            print(f"    [{a}, {b}] 宽 {b-a+1} px")
    print("col 540-660 每 10px（7px 均值）:")
    print("   ", {c: round(float(r[c-3:c+4].mean()), 3) for c in range(540, 661, 10)})
    print("col 300-380 每 10px（7px 均值）:")
    print("   ", {c: round(float(r[c-3:c+4].mean()), 3) for c in range(300, 381, 10)})
    print("col 820-880 每 10px（7px 均值）:")
    print("   ", {c: round(float(r[c-3:c+4].mean()), 3) for c in range(820, 881, 10)})


if __name__ == "__main__":
    main()
