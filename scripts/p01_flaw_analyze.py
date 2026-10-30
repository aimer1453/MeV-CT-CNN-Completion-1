"""P0-1 前置检查第三步（d）：缺陷的精确位置、尺寸与类型判定。

基于第三步（c）的 diff.npy / recon_flaw.npy / recon_unif.npy：
1. 9 根棒中心 + 间隙位置的差分值（判断缺陷在哪根棒/是否整棒）
2. 负差分区的连通域分析（缺陷尺寸）
3. 累加正弦图差分的重建（缺陷单体像，FBP 条纹少）
4. 正弦图轨迹的稳健峰值跟踪（独立定位）
5. 噪声底：正弦图差分在缺陷轨迹以外的统计（判断两扫描是否共用种子）
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "3_3_geant4"))
sys.path.insert(0, str(ROOT / "scripts"))

from quick_reconstruct import ring_correction, reconstruct_fbp  # noqa: E402

OUT = ROOT / "tmp_p01_flaw"
FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"

PITCH_CM = 2.6
PX_CM = 0.01
CENTER = 600
M = 1 + 31.7 / 55.0


def disk(cx, cy, r):
    y, x = np.ogrid[:1200, :1200]
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2


def main():
    recon_f = np.load(OUT / "recon_flaw.npy")
    recon_u = np.load(OUT / "recon_unif.npy")
    diff = recon_f - recon_u

    print("=" * 74)
    print("1. 各棒位/间隙的差分均值（半径 80 px 圆盘，棒外径 100 px）")
    print("=" * 74)
    print(f"{'位置':>14} {'x,y (cm)':>16} {'mean(diff)':>12} {'min(diff)':>12}")
    for bi in (-1, 0, 1):
        for bj in (-1, 0, 1):
            c = CENTER + bi * PITCH_CM / PX_CM
            r = CENTER - bj * PITCH_CM / PX_CM
            m = disk(c, r, 80)
            print(f"棒({bi:+d},{bj:+d})   ({bi*PITCH_CM:+.1f},{bj*PITCH_CM:+.1f})   "
                  f"{diff[m].mean():>12.6f} {diff[m].min():>12.6f}")
    # 间隙（相邻棒中点）
    for name, gx, gy in [("间隙(+x)", 1.3, 0.0), ("间隙(+y)", 0.0, 1.3),
                          ("间隙(+x+y)", 1.3, 1.3), ("间隙(-x+y)", -1.3, 1.3)]:
        c = CENTER + gx * 100
        r = CENTER - gy * 100
        m = disk(c, r, 60)
        print(f"{name:>14} ({gx*PITCH_CM:+.1f},{gy*PITCH_CM:+.1f})   "
              f"{diff[m].mean():>12.6f} {diff[m].min():>12.6f}")
    # 中心
    m = disk(CENTER, CENTER, 80)
    print(f"{'中心(0,0)':>14} (+0.0,+0.0)   {diff[m].mean():>12.6f} {diff[m].min():>12.6f}")

    print()
    print("=" * 74)
    print("2. 负差分连通域（scipy label，8 连通）")
    print("=" * 74)
    from scipy import ndimage
    neg = diff < diff.min() * 0.3
    lab, n = ndimage.label(neg)
    sizes = ndimage.sum(neg, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    print(f"负区总像素 {neg.sum()}，连通域 {n} 个；前 5 大:")
    for k in order[:5]:
        m = lab == (k + 1)
        rs, cs = np.where(m)
        print(f"  域{k+1}: {int(sizes[k])} px，行 {rs.min()}-{rs.max()} 列 {cs.min()}-{cs.max()}，"
              f"质心 ({rs.mean():.0f},{cs.mean():.0f}) → "
              f"x={(cs.mean()-CENTER)*PX_CM:+.2f} y={(CENTER-rs.mean())*PX_CM:+.2f} cm")
    # 最大连通域的等效半径
    m = lab == (order[0] + 1)
    rs, cs = np.where(m)
    print(f"  最大域等效半径 ≈ {np.sqrt(sizes[order[0]]/np.pi):.0f} px = "
          f"{np.sqrt(sizes[order[0]]/np.pi)*PX_CM:.2f} cm（棒芯半径 0.75 cm）")

    print()
    print("=" * 74)
    print("3. 累加正弦图差分的重建（缺陷单体像）")
    print("=" * 74)
    proj_f = {a: np.load(FLAW / f"projections/proj_a{a:06.1f}_z+0.npy").astype(np.float64)
              for a in range(360)}
    proj_u = {a: np.load(UNIF / f"projections/proj_a{a:03d}_z+0.npy").astype(np.float64)
              for a in range(360)}
    # 在 -log 域做差（与重建一致）：sinogram 已由 compute_sinogram_smooth_empty 生成过，
    # 这里直接对原始计数行做差再走 -log 会混入非线性；改用已存 sinogram 更稳——
    # 但 sinogram 未存，故用近似：对 flaw/unif 分别 -log 后相减（空场相同，可严格线性拆分）
    empty = np.load(FLAW / "results" / "empty_proj.npy").astype(np.float64)
    from quick_reconstruct import smooth_empty_proj
    e_row = np.maximum(smooth_empty_proj(empty, 2.0)[16], 1e-10)
    sino_diff = np.zeros((360, 1200))
    for a in range(360):
        lf = -np.log(np.maximum(proj_f[a][16], 1e-10) / e_row)
        lu = -np.log(np.maximum(proj_u[a][16], 1e-10) / e_row)
        sino_diff[a] = lf - lu
    sino_diff = ring_correction(sino_diff, method='fw', level=5)
    recon_def = reconstruct_fbp(sino_diff, list(range(360)))
    np.save(OUT / "recon_defect_only.npy", recon_def)
    print(f"缺陷单体像范围 [{recon_def.min():.6f}, {recon_def.max():.6f}]")
    i = np.unravel_index(np.argmin(recon_def), recon_def.shape)
    print(f"最小值 row={i[0]} col={i[1]} → x={(i[1]-CENTER)*PX_CM:+.2f} y={(CENTER-i[0])*PX_CM:+.2f} cm")
    m = recon_def < recon_def.min() * 0.5
    rs, cs = np.where(m)
    if len(rs):
        print(f"<50%最小值区域: {m.sum()} px，质心 ({rs.mean():.0f},{cs.mean():.0f}) → "
              f"x={(cs.mean()-CENTER)*PX_CM:+.2f} y={(CENTER-rs.mean())*PX_CM:+.2f} cm，"
              f"等效半径 {np.sqrt(m.sum()/np.pi)*PX_CM:.2f} cm")
    # 各棒位
    for bi in (-1, 0, 1):
        for bj in (-1, 0, 1):
            c = CENTER + bi * PITCH_CM / PX_CM
            r = CENTER - bj * PITCH_CM / PX_CM
            mm = disk(c, r, 80)
            print(f"  棒({bi:+d},{bj:+d}): mean={recon_def[mm].mean():.6f} min={recon_def[mm].min():.6f}")

    print()
    print("=" * 74)
    print("4. 正弦图轨迹稳健峰值跟踪（每角度平滑后取峰）")
    print("=" * 74)
    from scipy.ndimage import gaussian_filter1d
    cols, amps = [], []
    for a in range(360):
        p = gaussian_filter1d(sino_diff[a], 3)
        j = int(np.argmin(p))
        cols.append(j)
        amps.append(float(p[j]))
    cols = np.array(cols)
    print(f"峰值列范围 {cols.min()}-{cols.max()}；轨迹摆幅 ≈ {cols.max()-cols.min()} px "
          f"= {(cols.max()-cols.min())*PX_CM/M:.2f} cm（物体空间，放大率 {M:.3f}）")
    t = np.deg2rad(np.arange(360))
    A = np.c_[np.ones(360), np.cos(t), np.sin(t)]
    coef, *_ = np.linalg.lstsq(A, cols, rcond=None)
    amp = np.hypot(coef[1], coef[2])
    phi = np.arctan2(-coef[2], coef[1])
    rms = float(np.sqrt(np.mean((cols - A @ coef) ** 2)))
    print(f"正弦拟合 col = {coef[0]:.0f} + {amp:.0f}*cos(t{-np.rad2deg(phi):+.0f}°)，RMS={rms:.0f} px")
    print(f"⇒ 缺陷距中心 ≈ {amp*PX_CM/M:.2f} cm（{amp*PX_CM/M/PITCH_CM:.2f} 栅距）")

    print()
    print("=" * 74)
    print("5. 噪声底：正弦图差分在轨迹以外的统计")
    print("=" * 74)
    # 逐角度：差分曲线的 std（轨迹以外区域）
    stds = []
    for a in range(360):
        p = sino_diff[a].copy()
        j = cols[a]
        lo, hi = max(0, j - 30), min(1200, j + 30)
        mask = np.ones(1200, bool)
        mask[lo:hi] = False
        stds.append(p[mask].std())
    print(f"逐角度差分 std（轨迹 ±30 列以外）: 均值 {np.mean(stds):.6f}，最大 {np.max(stds):.6f}")
    print(f"对照：单角度 -log 值本身的典型 std ~ "
          f"{np.std(-np.log(np.maximum(proj_f[0][16],1e-10)/e_row)):.4f}")
    print("⇒ 若差分 std 远小于 -log 自身 std，说明两扫描噪声高度相关（共用种子/空场），")
    print("  差分图基本是纯缺陷信号")


if __name__ == "__main__":
    main()
