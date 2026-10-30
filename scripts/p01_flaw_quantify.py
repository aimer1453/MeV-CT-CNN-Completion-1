"""P0-1 第一步收尾：缺陷差异的全面定量（决定是否"差异不止一处"）。

已知：
- 缺陷棒 (i,j)=(-1,-1)，theta=0 时 (-2.6,-2.6) cm（flaw90 正弦拟合，amp=√2*260px）
- ring_correction(fw) 剥离中心列条纹 → 旧定位 (345,345) 是伪影
- b/c/d 路径一致：中心深谷 -0.01872（疑直流伪影）+ 4 边棒小团块 + R~368 弱环

本脚本定量：
1. 差分 sino 去均值后 FBP → 确认中心谷是直流伪影；凸显角棒环
2. R=368 环角向剖面 → 与真实缺陷方向 225° 对比，标定 recon 图像旋转偏差（第二步）
3. 9 棒格点逐棒采样（棒心 + 包壳环）→ 差异是否不止一处
4. 背景噪声基准 → 判断各差分的显著性

必须有 __main__ 保护。
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "3_3_geant4"))

from quick_reconstruct import reconstruct_fbp  # noqa: E402

FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"
OUT = ROOT / "tmp_p01_flaw"

angles = list(range(360))
CENTER = 600
PX = 0.01
PITCH_PX = 260  # 2.6 cm / 0.01


def build_dsino():
    """raw -log 差分 sino（flaw - unif），并去均值。"""
    d = np.zeros((360, 1200))
    for a in angles:
        pf = np.load(FLAW / "projections" / f"proj_a{a:06.1f}_z+0.npy").astype(np.float64)[16]
        pu = np.load(UNIF / "projections" / f"proj_a{a:03d}_z+0.npy").astype(np.float64)[16]
        ef = np.load(FLAW / "results" / "empty_proj.npy").astype(np.float64)[16]
        eu = np.load(UNIF / "results" / "empty_proj.npy").astype(np.float64)[16]
        d[a] = (-np.log(pf / ef + 1e-12)) - (-np.log(pu / eu + 1e-12))
    return d


def ring_profile(img, R, nbins=36):
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    rr = np.hypot(xx - CENTER, yy - CENTER)
    ang = np.degrees(np.arctan2(-(yy - CENTER), xx - CENTER)) % 360  # 图像极角
    prof = np.full(nbins, np.nan)
    for k in range(nbins):
        sel = (np.abs(rr - R) <= 1) & (ang >= k * 10) & (ang < (k + 1) * 10)
        if sel.sum():
            prof[k] = img[sel].mean()
    return prof


def main():
    print("构建 raw -log 差分 sino...", flush=True)
    d = build_dsino()
    np.save(OUT / "dsino_raw.npy", d)
    print(f"差分 sino: mean {d.mean():.3e}  std {d.std():.3e}  "
          f"min {d.min():.5f} max {d.max():.5f}")

    # ---- 1. 去直流 FBP ----
    print("\n[1] 去直流（整体减均值）后 FBP")
    d_dc = d - d.mean()
    img_dc = reconstruct_fbp(d_dc, angles)
    np.save(OUT / "diff_dc.npy", img_dc)
    print(f"  范围 [{img_dc.min():.5f}, {img_dc.max():.5f}] mean {img_dc.mean():.3e}")
    i = np.unravel_index(np.argmin(img_dc), img_dc.shape)
    print(f"  最小值: row={i[0]} col={i[1]} val={img_dc[i]:.5f}"
          f"（原直流版在中心 (600,600) val=-0.01872）")
    print(f"  中心值 now: {img_dc[600, 600]:.5f}（去直流后应接近 0）")

    # 连通域
    thr = img_dc.min() * 0.5
    mask = img_dc < thr
    lab, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    print(f"  连通域（<50%最小值）数: {n}，前 5 大:")
    for k in order[:5]:
        m = lab == (k + 1)
        rs, cs = np.where(m)
        print(f"    {int(sizes[k])} px 质心 (row={rs.mean():.0f}, col={cs.mean():.0f}) "
              f"距中心 {np.hypot(cs.mean()-CENTER, rs.mean()-CENTER):.0f} px")

    # ---- 2. 角棒环角向剖面 + 图像旋转标定 ----
    print("\n[2] R=368 环角向剖面（10° bin，图像极角，+x 为 0°，逆时针）")
    prof = ring_profile(img_dc, 368)
    for k in range(36):
        bar = "#" * int(max(0, -prof[k]) * 4000)
        print(f"    {k*10:3d}-{k*10+10:3d}°: {prof[k]:+.5f} {bar}")
    kmin = int(np.nanargmin(prof))
    print(f"  最强负差分方向（图像极角）: {kmin*10}-{kmin*10+10}°，值 {prof[kmin]:.5f}")
    print(f"  真实缺陷方向（几何推导，世界坐标 atan2(y,x)）: 225°")
    print(f"  → 图像相对世界的旋转偏差: {(kmin*10+5) - 225:+.0f}°")

    # ---- 3. 9 棒格点逐棒采样 ----
    print("\n[3] 9 棒格点采样（recon 图坐标，90° 旋转下不变的格点集合）")
    print("    棒心: 5x5 邻域均值；包壳环: 棒心外 78px 处 8 方向 3x3 均值")
    rods = {}
    for bi in (-1, 0, 1):
        for bj in (-1, 0, 1):
            rods[(bi, bj)] = (CENTER + bi * PITCH_PX, CENTER - bj * PITCH_PX)  # (col, row)
    print(f"    {'棒(i,j)':>10} {'图(col,row)':>16} {'棒心差分':>10} {'包壳环差分':>12}")
    for (bi, bj), (c, r) in rods.items():
        core = img_dc[r-2:r+3, c-2:c+3].mean()
        shell_vals = []
        for angd in range(0, 360, 45):
            dr, dc = int(78 * np.sin(np.radians(angd))), int(78 * np.cos(np.radians(angd)))
            rr2, cc2 = r + dr, c + dc
            shell_vals.append(img_dc[rr2-1:rr2+2, cc2-1:cc2+2].mean())
        shell = float(np.mean(shell_vals))
        print(f"    ({bi:+d},{bj:+d})   ({c:4d},{r:4d})   {core:+.5f}   {shell:+.5f}")

    # ---- 4. 背景噪声 ----
    print("\n[4] 背景噪声基准")
    yy, xx = np.ogrid[:img_dc.shape[0], :img_dc.shape[1]]
    rr = np.hypot(xx - CENTER, yy - CENTER)
    for R in (430, 470, 510):
        sel = np.abs(rr - R) <= 2
        print(f"    R={R} 环: mean {img_dc[sel].mean():+.6f}  std {img_dc[sel].std():.6f}  "
              f"n={sel.sum()}")

    # ---- 5. 投影域直接验证：缺陷棒芯 vs 正常棒芯的 -log 差分 ----
    print("\n[5] 投影域验证（raw counts，缺陷棒 vs 对角正常棒，跨全部 360 角）")
    # 缺陷棒 theta=0 在 (-2.6,-2.6)；其轨迹列 col(a) = 600 + 260*cos(a+225)
    # 对角正常棒 (i=+1,j=-1) theta=0 在 (+2.6,-2.6)，轨迹 col(a) = 600 + 260*cos(a+315)
    gains, refs = [], []
    for a in angles:
        t = np.radians(a)
        c_def = int(round(CENTER + 368.2 * np.cos(t + np.radians(225))))
        c_ref = int(round(CENTER + 368.2 * np.cos(t + np.radians(315))))
        pf = np.load(FLAW / "projections" / f"proj_a{a:06.1f}_z+0.npy").astype(np.float64)[16]
        pu = np.load(UNIF / "projections" / f"proj_a{a:03d}_z+0.npy").astype(np.float64)[16]
        # 缺陷棒芯 3px 邻域（flaw vs unif 在同列同物理点）
        gains.append(pf[c_def-1:c_def+2].sum() / pu[c_def-1:c_def+2].sum())
        refs.append(pf[c_ref-1:c_ref+2].sum() / pu[c_ref-1:c_ref+2].sum())
    gains = np.array(gains)
    refs = np.array(refs)
    print(f"    缺陷棒芯计数比 flaw/unif: mean {gains.mean():.4f}  std {gains.std():.4f}  "
          f"min {gains.min():.4f} max {gains.max():.4f}")
    print(f"    对角正常棒芯计数比:       mean {refs.mean():.4f}  std {refs.std():.4f}")
    print(f"    → 缺陷棒透射增强 {(gains.mean()-1)*100:+.2f}%（正常棒 {(refs.mean()-1)*100:+.2f}%）")


if __name__ == "__main__":
    main()
