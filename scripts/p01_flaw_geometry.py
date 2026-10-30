"""P0-1 前置检查第三步（g）：修正几何（放大率）后重新归因。

第三步（f）发现：per-rod 测量位置可能错（未计放大率 1.576）。本脚本：
1. 从 recon_unif 径向轮廓实测棒位（不依赖假设）
2. 从 proj[0] row16 峰值实测探测器上的棒投影位置
3. 用实测棒位重做逐棒差分
4. 检查均匀 3.0MeV 宏文件的能量（排除能量不一致）
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tmp_p01_flaw"
FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"
CENTER = 600


def main():
    recon_u = np.load(OUT / "recon_unif.npy")
    recon_f = np.load(OUT / "recon_flaw.npy")
    diff = recon_f - recon_u

    # ---------- 1. 径向轮廓实测棒位 ----------
    print("=" * 74)
    print("1. recon_unif 径向轮廓（绕 (600,600)，找棒位峰）")
    print("=" * 74)
    y, x = np.ogrid[:1200, :1200]
    r = np.sqrt((x - CENTER) ** 2 + (y - CENTER) ** 2)
    prof = np.zeros(601)
    cnt = np.zeros(601)
    ri = r.astype(int).ravel()
    prof = np.bincount(ri, weights=recon_u.ravel(), minlength=601)
    cnt = np.bincount(ri, minlength=601).astype(float)
    cnt[cnt == 0] = 1
    prof = prof / cnt
    # 打印 r=0..600 每 20 px 的均值，标出局部峰
    print("  r(px)  mean")
    for rr in range(0, 601, 20):
        bar = "#" * int(prof[rr] / prof[3:600].max() * 50)
        print(f"  {rr:4d}  {prof[rr]:.5f} {bar}")
    peaks = [i for i in range(5, 596) if prof[i] == max(prof[i - 5:i + 6]) and prof[i] > 0.3 * prof.max()]
    print(f"  局部峰位置: {peaks}")

    # ---------- 2. 投影峰值实测 ----------
    print()
    print("=" * 74)
    print("2. proj[0] row16 原始计数峰（探测器上的棒投影位置）")
    print("=" * 74)
    for tag, path in [("flaw", FLAW / "projections/proj_a0000.0_z+0.npy"),
                      ("unif", UNIF / "projections/proj_a000_z+0.npy")]:
        p = np.load(path)[16].astype(np.float64)
        from scipy.ndimage import gaussian_filter1d
        ps = gaussian_filter1d(p, 5)
        pk = [i for i in range(3, 1197) if ps[i] == max(ps[i - 3:i + 4]) and ps[i] > 0.5 * ps.max()]
        print(f"  {tag}: 峰列 {pk}；峰谷: 峰均值 {ps[pk].mean():.1f} / 全均值 {ps.mean():.1f}")
        print(f"    计数范围 [{p.min():.0f}, {p.max():.0f}]，全列和 {p.sum():.0f}")

    # ---------- 3. 实测棒位逐棒差分 ----------
    print()
    print("=" * 74)
    print("3. 用实测棒位重做逐棒差分（棒芯 r<=60，环 75<=r<=95）")
    print("=" * 74)
    def disk(cx, cy, rad):
        yy, xx = np.ogrid[:1200, :1200]
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
    # 棒位：中心 (600,600)；8 邻棒在半径 R 的圆上（R 由第 1 步实测）
    R = peaks[1] if len(peaks) > 1 else 410
    print(f"  实测邻棒半径 R = {R} px")
    rods = [(0, 0, CENTER, CENTER)]
    for bi in (-1, 0, 1):
        for bj in (-1, 0, 1):
            if bi == 0 and bj == 0:
                continue
            rods.append((bi, bj, CENTER + bi * R, CENTER - bj * R))
    print(f"{'棒位':>8} {'图像(col,row)':>16} {'flaw芯':>9} {'unif芯':>9} {'Δ芯':>10} "
          f"{'flaw环':>9} {'unif环':>9} {'Δ环':>10}")
    for bi, bj, c, rr_ in rods:
        core = disk(c, rr_, 60)
        ring = disk(c, rr_, 95) & ~disk(c, rr_, 75)
        print(f"({bi:+d},{bj:+d}) ({c:4d},{rr_:4d})  {recon_f[core].mean():>9.5f} "
              f"{recon_u[core].mean():>9.5f} {recon_f[core].mean()-recon_u[core].mean():>10.5f} "
              f"{recon_f[ring].mean():>9.5f} {recon_u[ring].mean():>9.5f} "
              f"{recon_f[ring].mean()-recon_u[ring].mean():>10.5f}")

    # 棒间介质（相邻棒中点）
    print()
    print("  棒间介质（相邻棒连线中点，r<=40）:")
    for bi, bj, c, rr_ in rods[1:5]:
        # 取该棒与中心棒连线的中点
        mc, mr = (c + CENTER) // 2, (rr_ + CENTER) // 2
        m = disk(mc, mr, 40)
        print(f"    棒({bi:+d},{bj:+d})-中心 中点 ({mc},{mr}): "
              f"flaw={recon_f[m].mean():.5f} unif={recon_u[m].mean():.5f} "
              f"Δ={recon_f[m].mean()-recon_u[m].mean():+.5f}")

    # ---------- 4. 均匀宏文件能量核对 ----------
    print()
    print("=" * 74)
    print("4. 宏文件能量核对（排除两扫描能量不一致）")
    print("=" * 74)
    import re
    for tag, d in [("flaw", FLAW), ("unif", UNIF)]:
        macs = sorted((d / "macros").glob("*.mac"))
        vals = set()
        for m in macs:
            txt = m.read_text(errors="ignore")
            mm = re.search(r"/gps/ene/mono\s+([\d.]+)\s*MeV", txt)
            if mm:
                vals.add(mm.group(1))
        print(f"  {tag}: {len(macs)} 个宏，能量取值 {vals}")


if __name__ == "__main__":
    main()
