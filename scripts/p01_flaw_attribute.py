"""P0-1 前置检查第三步（f）：差分模式的归因（几何配准 vs 材质差异）。

第三步（e）的 ASCII 图显示全部 9 根棒都有环形差分 + 全局负偏置。本脚本判别：
  (b) 几何配准不一致（棒位/旋转中心/放大率偏移）→ 每根棒边缘出现正负成对
  (c) 材质差异（密度/半径/空洞）→ 棒芯整体偏移，无正负成对
做法：
  1. 两重建图互相关求最佳平移
  2. 逐棒测量：棒心位置（质心）、棒芯 mean、棒芯 min、边缘环均值
  3. 全图总衰减（sinogram 求和）对比
  4. 正弦图差分 ASCII + 轨迹
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "3_3_geant4"))
sys.path.insert(0, str(ROOT / "scripts"))

from quick_reconstruct import smooth_empty_proj  # noqa: E402

OUT = ROOT / "tmp_p01_flaw"
FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"
CENTER = 600
PX_CM = 0.01


def disk(cx, cy, r):
    y, x = np.ogrid[:1200, :1200]
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2


def main():
    recon_f = np.load(OUT / "recon_flaw.npy")
    recon_u = np.load(OUT / "recon_unif.npy")
    diff = recon_f - recon_u

    # ---------- 1. 互相关求平移 ----------
    print("=" * 74)
    print("1. 重建图互相关（找最佳整数平移）")
    print("=" * 74)
    best = None
    for dy in range(-12, 13):
        for dx in range(-12, 13):
            a = recon_f[max(0, dy):1200 + min(0, dy), max(0, dx):1200 + min(0, dx)]
            b = recon_u[max(0, -dy):1200 + min(0, -dy), max(0, -dx):1200 + min(0, -dx)]
            v = float(np.mean(a * b))
            if best is None or v > best[0]:
                best = (v, dy, dx)
    print(f"  最佳平移 dy={best[1]}, dx={best[2]}（相关 {best[0]:.3e}）")
    # 平移后的残差
    dy, dx = best[1], best[2]
    a = recon_f[max(0, dy):1200 + min(0, dy), max(0, dx):1200 + min(0, dx)]
    b = recon_u[max(0, -dy):1200 + min(0, -dy), max(0, -dx):1200 + min(0, -dx)]
    print(f"  平移前 mean(diff^2) = {np.mean(diff**2):.3e}；"
          f"平移后 = {np.mean((a-b)**2):.3e}")

    # ---------- 2. 逐棒测量 ----------
    print()
    print("=" * 74)
    print("2. 逐棒测量（棒芯 r<=60 px；环 75<=r<=95）")
    print("=" * 74)
    print(f"{'棒位':>10} {'flaw芯mean':>11} {'unif芯mean':>11} {'Δ芯':>11} "
          f"{'flaw环mean':>11} {'unif环mean':>11} {'Δ环':>11}")
    for bi in (-1, 0, 1):
        for bj in (-1, 0, 1):
            c = CENTER + bi * 260
            r = CENTER - bj * 260
            core = disk(c, r, 60)
            ring = disk(c, r, 95) & ~disk(c, r, 75)
            print(f"({bi:+d},{bj:+d})  {recon_f[core].mean():>11.5f} {recon_u[core].mean():>11.5f} "
                  f"{recon_f[core].mean()-recon_u[core].mean():>11.5f} "
                  f"{recon_f[ring].mean():>11.5f} {recon_u[ring].mean():>11.5f} "
                  f"{recon_f[ring].mean()-recon_u[ring].mean():>11.5f}")

    # ---------- 3. 总衰减对比 ----------
    print()
    print("=" * 74)
    print("3. 总衰减（row16 sinogram 求和）与平均计数")
    print("=" * 74)
    empty = np.load(FLAW / "results" / "empty_proj.npy").astype(np.float64)
    e_row = np.maximum(smooth_empty_proj(empty, 2.0)[16], 1e-10)
    sf = su = cf = cu = 0.0
    for a in range(360):
        pf = np.load(FLAW / f"projections/proj_a{a:06.1f}_z+0.npy")[16].astype(np.float64)
        pu = np.load(UNIF / f"projections/proj_a{a:03d}_z+0.npy")[16].astype(np.float64)
        sf += float(np.sum(-np.log(np.maximum(pf, 1e-10) / e_row)))
        su += float(np.sum(-np.log(np.maximum(pu, 1e-10) / e_row)))
        cf += float(pf.sum()); cu += float(pu.sum())
    print(f"  flaw: 360 角 -log 总和 = {sf:.1f}；总计数 = {cf:.3e}")
    print(f"  unif: 360 角 -log 总和 = {su:.1f}；总计数 = {cu:.3e}")
    print(f"  Δ(-log 总和) = {sf-su:+.1f} ({(sf-su)/su*100:+.3f}%)；"
          f"Δ(总计数) = {(cf-cu)/cu*100:+.3f}%")

    # ---------- 4. 正弦图差分 ASCII + 轨迹 ----------
    print()
    print("=" * 74)
    print("4. 正弦图差分 ASCII（36x60；' '最强负=缺陷迹）")
    print("=" * 74)
    sino_diff = np.zeros((360, 1200))
    for a in range(360):
        pf = np.load(FLAW / f"projections/proj_a{a:06.1f}_z+0.npy")[16].astype(np.float64)
        pu = np.load(UNIF / f"projections/proj_a{a:03d}_z+0.npy")[16].astype(np.float64)
        sino_diff[a] = (-np.log(np.maximum(pf, 1e-10) / e_row)
                        - -np.log(np.maximum(pu, 1e-10) / e_row))
    chars = " .:-=+*#%@"
    g2 = sino_diff.reshape(36, 10, 60, 20).mean(axis=(1, 3))
    lo2, hi2 = np.percentile(g2, [1, 99])
    for row in range(36):
        line = ""
        for col in range(60):
            v = (g2[row, col] - lo2) / (hi2 - lo2 + 1e-12)
            k = int(np.clip(v * (len(chars) - 1), 0, len(chars) - 1))
            line += chars[k]
        print(f"   a={row*10:3d} {line}")
    print(f"   （色窗 [{lo2:.5f}, {hi2:.5f}]；列 30 = 探测器中心 600）")

    # 每角度最小值位置（缺陷迹）
    from scipy.ndimage import gaussian_filter1d
    tr = [int(np.argmin(gaussian_filter1d(sino_diff[a], 3))) for a in range(360)]
    tr = np.array(tr)
    print(f"   逐角最小列: 范围 {tr.min()}-{tr.max()}，围绕 {np.median(tr)}")
    # 直方图
    hist, edges = np.histogram(tr, bins=12, range=(0, 1200))
    print(f"   列直方图(12 bin): {hist.tolist()}")


if __name__ == "__main__":
    main()
