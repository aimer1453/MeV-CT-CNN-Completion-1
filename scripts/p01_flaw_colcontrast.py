"""补充对照：32 行累加口径下，各棒列的系统偏移与缺陷棒列的额外信号。

目的：区分「共同偏移（RNG 漂移）」与「缺陷额外信号」。
- col 340 = 缺陷棒 (-1,+1) 在 a=0 的投影列（3 棒叠加）
- col 600 = 中心棒 + 2 边棒（a=0）
- col 860 = 3 根正常角棒（a=0）
- col 1100 = 纯背景
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"
angles = list(range(360))


def main():
    pf = np.stack([np.load(FLAW / "projections" / f"proj_a{a:06.1f}_z+0.npy").astype(np.float64)
                   for a in angles]).sum(axis=1)
    pu = np.stack([np.load(UNIF / "projections" / f"proj_a{a:03d}_z+0.npy").astype(np.float64)
                   for a in angles]).sum(axis=1)
    d = np.log(np.where(pu > 0, pu, 1.0)) - np.log(np.where(pf > 0, pf, 1.0))

    print("32 行累加口径，每列 d = log(unif) - log(flaw)（正值 = flaw 计数更多）")
    print(f"{'列':>6} {'含义':<26} {'mean d':>10} {'sem':>8} {'vs col860':>10}")
    ref = d[:, 858:863].mean()
    rows = [
        (340, "缺陷棒列 (-1,+1) a=0"),
        (600, "中心棒+2边棒 a=0"),
        (860, "3 正常角棒 a=0"),
        (260, "边棒列 a=0 (x=-3.4?)"),
        (940, "边棒列 a=0"),
        (1100, "纯背景"),
    ]
    for c, desc in rows:
        seg = d[:, c - 2:c + 3]
        m, sem = float(np.mean(seg)), float(np.std(seg) / np.sqrt(seg.size))
        print(f"{c:6d} {desc:<26} {m:+10.4f} {sem:8.4f} {m - ref:+10.4f}")

    # 缺陷棒列 - 正常角棒列 的差值（消除共同偏移）
    print("\n差分（扣除共同偏移，col340 - col860）:")
    diff = d[:, 338:343].mean(axis=1) - d[:, 858:863].mean(axis=1)
    print(f"  逐角度 mean {diff.mean():+.4f}  std {diff.std():.4f}  "
          f"sem {diff.std()/np.sqrt(360):.4f}")
    print(f"  符号一致性 d>0: {int((diff > 0).sum())}/360 "
          f"({(diff > 0).sum()/360*100:.1f}%)")

    # 缺陷棒 vs 中心棒
    print("\n差分（col340 - col600，缺陷棒 vs 中心棒）:")
    diff2 = d[:, 338:343].mean(axis=1) - d[:, 598:603].mean(axis=1)
    print(f"  mean {diff2.mean():+.4f}  std {diff2.std():.4f}  "
          f"sem {diff2.std()/np.sqrt(360):.4f}")


if __name__ == "__main__":
    main()
