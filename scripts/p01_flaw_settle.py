"""P0-1 收尾验证：角度映射标定 + 中心列条纹归因（RNG 漂移 vs 第二缺陷）。

背景（用户 2026-10-23 22:30 裁决）：
- 缺陷版代码只有一处差异：`(i == -1 && j == 1) ? SolidFuelWithBubble : SolidFuelBase`
  → 单缺陷，世界坐标 (-2.6, +2.6) cm（世界极角 135°，第二象限）
- 我实测 flaw90 拟合 col(a) = 601.9 + 367.1*cos(a + 224.7°)
- 需对比代码坐标的正确投影式，标定 flaw 版 scan.py 的角度映射

本脚本：
1. 三候选公式 vs 实测列的 RMS 残差 + 相位差
   - C_code: col = 600 + 367*cos(a + 135°)   [(-2.6,+2.6) 逆时针无偏移]
   - C_user: col = 600 + 367*cos(a - 135°)   [用户给的式子，注意符号]
   - C_meas: 601.9 + 367.1*cos(a + 224.7°)   [实测]
2. 中心列条纹归因：
   - 32 行累加（降泊松噪声 √32）
   - 绝对计数水平（判断低计数区泊松涨落量级）
   - 空间剖面 5px 分箱（平滑=整棒系统差；尖峰=单列异常）
   - 角度稳定性 mean/std
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"
angles = list(range(360))


def load():
    pf = np.stack([np.load(FLAW / "projections" / f"proj_a{a:06.1f}_z+0.npy").astype(np.float64)
                   for a in angles])
    pu = np.stack([np.load(UNIF / "projections" / f"proj_a{a:03d}_z+0.npy").astype(np.float64)
                   for a in angles])
    return pf, pu


def part1_angle_mapping(pf):
    print("=" * 76)
    print("[1] 角度映射标定：代码坐标 (-2.6,+2.6) vs 实测正弦轨迹")
    print("=" * 76)
    # 复算 flaw90 argmax 轨迹（与 p01_flaw_relocalize.py 同法）
    cols = np.array([int(np.argmax(pf[a][16] - pf[(a + 90) % 360][16])) for a in angles],
                    dtype=float)
    t = np.deg2rad(np.arange(360, dtype=float))
    A = np.c_[np.ones(360), np.cos(t), np.sin(t)]
    coef, *_ = np.linalg.lstsq(A, cols, rcond=None)
    c0, Ac, Bs = coef
    amp = float(np.hypot(Ac, Bs))
    phi = float(np.degrees(np.arctan2(-Bs, Ac)) % 360)
    pred = A @ coef
    rms = float(np.sqrt(np.mean((cols - pred) ** 2)))
    print(f"  实测: col(a) = {c0:.1f} + {amp:.1f}*cos(a + {phi:.1f}°)，RMS {rms:.1f} px")
    print(f"        → 该式等价于 (x0,y0) = ({amp*np.cos(np.radians(phi))/100:+.2f}, "
          f"{amp*np.sin(np.radians(phi))/100:+.2f}) cm，极角 {phi:.1f}°")

    # 三候选
    r = amp
    cands = {
        "C_code 代码坐标(-2.6,+2.6)+逆时针无偏移: 600 + 367cos(a+135°)":
            600 + r * np.cos(t + np.radians(135)),
        "C_user 用户式子: 600 + 367cos(a-135°)":
            600 + r * np.cos(t - np.radians(135)),
        "C_meas 实测: 601.9 + 367.1cos(a+224.7°)":
            c0 + amp * np.cos(t + np.radians(phi)),
    }
    print()
    for name, c in cands.items():
        print(f"  {name}")
        print(f"    RMS vs 实测列: {np.sqrt(np.mean((c - cols)**2)):.1f} px；"
              f"a=0 列 {c[0]:.0f}，a=90 列 {c[90]:.0f}，a=180 列 {c[180]:.0f}")

    print()
    print(f"  相位差: 实测 {phi:.1f}° − 代码坐标正确式 135° = {phi - 135:+.1f}°")
    print(f"  ⇒ 若 flaw 版 rotationAngle = 文件名角度 + 90°，则 (-2.6,+2.6) 的轨迹为")
    print(f"     600 + 367cos(a + 225°)，与实测 {phi:.1f}° 差 {phi - 225:+.1f}°（吻合）")
    print("  ⇒ 等价候选：flaw 版用顺时针 rotationAngle = −a，同样给出 cos(a+225°)")
    print("  ⚠ 90° 整数倍偏移在投影域不可区分（4 根角棒在 D4 下等价，同一 logical volume）")
    print("     → 缺陷棒具体是哪根只能以代码为准：(i,j) = (-1,+1)")


def part2_centre_stripe(pf, pu):
    print()
    print("=" * 76)
    print("[2] 中心列条纹归因：中心棒 RNG 漂移 vs 旋转轴第二缺陷")
    print("=" * 76)
    # 绝对计数水平（unif，32 行累加前先看单行）
    print("  绝对计数水平（unif a000 row16，col 540-660 每 20px 均值）:")
    u = pu[0, 16]
    for c0 in range(540, 661, 20):
        print(f"    col {c0}: {u[c0-5:c0+6].mean():.2f} counts/px"
              f"（-log 域 {(-np.log(max(u[c0-5:c0+6].mean(), 1e-9)/22.5)):.2f}）")

    # 32 行累加：d(a,c) = -log(f) - (-log(u))，仅用两扫描都 >0 的像素
    cf = pf.sum(axis=1)   # (360, 1200) 行累加
    cu = pu.sum(axis=1)
    valid = (cf > 0) & (cu > 0)
    d = np.where(valid, -np.log(np.where(cf > 0, cf, 1.0)) + np.log(np.where(cu > 0, cu, 1.0)), np.nan)
    # 注: -log(cf) + log(cu) = log(cu/cf) = -(log(cf/cu))

    print("\n  32 行累加后 col 500-700 的差分（5px 分箱）:")
    print(f"    {'col':>6} {'mean d':>10} {'std d':>10} {'sem':>9} {'n_valid':>8}")
    for c0 in range(500, 701, 10):
        sl = slice(c0 - 2, c0 + 3)
        dd = d[:, sl]
        m = np.nanmean(dd)
        s = np.nanstd(dd)
        nv = int(np.sum(~np.isnan(dd)))
        sem = s / np.sqrt(max(nv, 1))
        print(f"    {c0:6d} {m:+10.4f} {s:10.4f} {sem:9.4f} {nv:8d}")

    # 中心列 vs 邻列的显著性与角度稳定性
    print("\n  角度稳定性（col 600 单列，32 行累加）:")
    col = 600
    dd = d[:, col]
    ok = ~np.isnan(dd)
    print(f"    mean {np.nanmean(dd):+.4f}  std {np.nanstd(dd):.4f}  "
          f"sem {np.nanstd(dd)/np.sqrt(ok.sum()):.4f}  n {ok.sum()}/360")
    neg = np.sum(dd[ok] > 0)
    print(f"    符号一致性: {neg}/{ok.sum()} 个角度 d>0（{neg/ok.sum()*100:.1f}%）")
    print("    → 若 mean >> sem 且符号高度一致 = 系统差；若 std 远大于 mean = 纯涨落")

    # 关键判据：整棒宽度（±100px）内的平均 vs 单列
    print("\n  整棒宽度平均 vs 单列（32 行累加，a 全部）:")
    for name, sl in [("col 600±1", slice(599, 602)),
                     ("col 600±3", slice(597, 604)),
                     ("col 600±10", slice(590, 611)),
                     ("col 600±50", slice(550, 651)),
                     ("col 600±100", slice(500, 701))]:
        seg = d[:, sl]
        print(f"    {name:14s}: mean {np.nanmean(seg):+.5f}  "
              f"sem {np.nanstd(seg)/np.sqrt(max(np.sum(~np.isnan(seg)), 1)):.5f}")

    # 对照：正常角棒列（col 860）与纯背景列（col 1100）
    print("\n  对照:")
    for name, col in [("col 860 (3 正常角棒)", 860), ("col 1100 (纯背景)", 1100)]:
        dd = d[:, col]
        print(f"    {name}: mean {np.nanmean(dd):+.5f}  "
              f"std {np.nanstd(dd):.5f}")


def main():
    pf, pu = load()
    part1_angle_mapping(pf)
    part2_centre_stripe(pf, pu)


if __name__ == "__main__":
    main()
