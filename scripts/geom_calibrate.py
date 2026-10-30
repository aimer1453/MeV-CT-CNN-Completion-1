"""几何标定（第一步/第二步共用）：实测正弦图棒迹位置，裁定放大率与轴映射。

代码事实（DetectorConstruction.cc/.hh）：
- 源 (0, +55, 0) cm，方向 (0,-1,0)；探测器 y = -4.5*2.6 - 20 = -31.7 cm
- 传感器 1200(x) × 32(z)，像素 0.01 cm；i 从 -600..599，sensorX=(i+0.5)*0.01
- copyNo = (j+16)*1200 + (i+600) → 列索引 = i+600，sensorX 随 i 增大
- 组件绕 z 转 rotationAngle（FUEL_ROTATION_ANGLE 环境变量）
- 几何放大率 M = (55+31.7)/55 = 1.576

待裁定：棒（物空间 ±2.6 cm）在正弦图/图像里是 ±410 px（含放大）还是 ±260 px（1:1）。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"


def load(scan, a, naming):
    if naming == "float":
        return np.load(scan / f"projections/proj_a{a:06.1f}_z+0.npy")[16].astype(np.float64)
    return np.load(scan / f"projections/proj_a{a:03d}_z+0.npy")[16].astype(np.float64)


def main():
    print("=" * 74)
    print("1. 角度 0 的 row16 计数剖面（每 20 列打印，找棒吸收谷）")
    print("=" * 74)
    for tag, scan, naming in [("flaw", FLAW, "float"), ("unif", UNIF, "int")]:
        p = load(scan, 0, naming)
        print(f"  [{tag}] 每 20 列均值:")
        line = "    "
        for c in range(0, 1200, 20):
            v = p[c:c + 20].mean()
            line += f"{v:5.1f}"
            if (c // 20 + 1) % 10 == 0:
                print(line)
                line = "    "
        if line.strip():
            print(line)
        # 平滑后找谷
        from scipy.ndimage import gaussian_filter1d
        ps = gaussian_filter1d(p, 8)
        mins = [c for c in range(30, 1170)
                if ps[c] == min(ps[max(0, c - 60):c + 61]) and ps[c] < 0.6 * ps.mean()]
        print(f"  [{tag}] 谷位（棒投影列）: {mins}")

    print()
    print("=" * 74)
    print("2. 全角度棒迹幅度（正弦图逐角谷位直方图）")
    print("=" * 74)
    from scipy.ndimage import gaussian_filter1d
    for tag, scan, naming in [("flaw", FLAW, "float"), ("unif", UNIF, "int")]:
        allmins = []
        for a in range(0, 360, 5):
            p = load(scan, a, naming)
            ps = gaussian_filter1d(p, 8)
            mins = [c for c in range(30, 1170)
                    if ps[c] == min(ps[max(0, c - 60):c + 61]) and ps[c] < 0.6 * ps.mean()]
            allmins.extend(mins)
        allmins = np.array(allmins)
        hist, edges = np.histogram(allmins, bins=24, range=(0, 1200))
        print(f"  [{tag}] 谷位列直方图（24 bin，每 bin 50 px）:")
        for i, h in enumerate(hist):
            bar = "#" * h
            print(f"    {int(edges[i]):4d}-{int(edges[i+1]):4d}: {bar} ({h})")
        print(f"    非零谷位距 600 的 |偏移| 中位数: "
              f"{np.median(np.abs(allmins - 600)):.0f} px")

    print()
    print("=" * 74)
    print("3. 结论对照")
    print("=" * 74)
    print("  若谷位集中在 ±260 px → 正弦图是物空间 1:1（无放大）")
    print("  若谷位集中在 ±390-430 px → 含几何放大 1.576")


if __name__ == "__main__":
    main()
