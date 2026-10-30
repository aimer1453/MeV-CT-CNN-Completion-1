"""P0-1 收尾：缺陷几何形状测量（从数据反推仿真意图）。

用去直流差分 recon（diff_dc.npy）+ 投影域列比值，测量：
1. 中心棒差分团：尺寸/形状（整根棒芯 vs 局部气泡）
2. 角棒缺陷：在 R=368 环上的角向宽度（局部 vs 整根棒）
3. 缺陷的 z 方向范围（32 行中哪些行有差分）
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "3_3_geant4"))
from quick_reconstruct import reconstruct_fbp  # noqa: E402

OUT = ROOT / "tmp_p01_flaw"
CENTER = 600


def main():
    img = np.load(OUT / "diff_dc.npy")
    print(f"差分图（去直流）: [{img.min():.5f}, {img.max():.5f}]")

    # ---- 1. 中心棒差分团 ----
    print("\n[1] 中心棒区域（(500,500)-(700,700)）差分团")
    sub = img[500:701, 500:701]
    thr = sub.min() * 0.5
    mask = sub < thr
    lab, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    m = lab == k
    rs, cs = np.where(m)
    print(f"    最强连通域 {int(sizes[k-1])} px，row {rs.min()+500}-{rs.max()+500}，"
          f"col {cs.min()+500}-{cs.max()+500}")
    print(f"    尺寸: {rs.max()-rs.min()+1} x {cs.max()-cs.min()+1} px"
          f"（棒芯直径 150 px，包壳 200 px）")
    print(f"    等效半径 {np.sqrt(sizes[k-1]/np.pi):.0f} px（燃料半径 75 px）")
    # 形状：各向异性
    if len(rs) > 10:
        cov = np.cov(np.c_[rs, cs].T)
        w, v = np.linalg.eigh(cov)
        print(f"    主轴/次轴比 {np.sqrt(w[1]/w[0]):.2f}（1.0 = 圆形）")
    # 差分幅度沿半径
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    rr = np.hypot(xx - CENTER, yy - CENTER)
    for R in (20, 40, 60, 75, 90, 110):
        sel = np.abs(rr - R) <= 2
        print(f"    R={R:3d} px 环均值: {img[sel].mean():+.5f}")

    # ---- 2. 角棒缺陷弧宽 ----
    print("\n[2] 角棒缺陷在 R=368 环上的角向分布（5° bin）")
    ang = np.degrees(np.arctan2(-(yy - CENTER), xx - CENTER)) % 360
    prof = np.full(72, np.nan)
    for k in range(72):
        sel = (np.abs(rr - 368) <= 2) & (ang >= k * 5) & (ang < (k + 1) * 5)
        if sel.sum():
            prof[k] = img[sel].mean()
    kmin = int(np.nanargmin(prof))
    print(f"    峰在 {kmin*5}-{kmin*5+5}°（值 {prof[kmin]:.5f}）")
    half = prof < prof[kmin] * 0.5
    ks = np.where(half)[0]
    print(f"    <50% 峰的 bin: {[int(k*5) for k in ks]} → 弧宽约 {len(ks)*5}°")
    print(f"    （整根棒投影张角 ~200px/368px ≈ 31°；局部缺陷应更窄）")
    for k in range(24, 32):
        print(f"      {k*5:3d}-{k*5+5:3d}°: {prof[k]:+.5f} {'#'*int(max(0,-prof[k])*3000)}")

    # ---- 3. z 方向范围（投影域，用列比值）----
    print("\n[3] 缺陷的 z（行）方向范围：a000/a090/a180/a270 的列 600 与 340 比值按行")
    FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
    UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"
    for tag, fa, ua in [("a000", "0000.0", "000"), ("a090", "0090.0", "090"),
                        ("a180", "0180.0", "180"), ("a270", "0270.0", "270")]:
        pf = np.load(FLAW / "projections" / f"proj_a{fa}_z+0.npy").astype(np.float64)
        pu = np.load(UNIF / "projections" / f"proj_a{ua}_z+0.npy").astype(np.float64)
        r600 = pf[:, 598:603].sum(1) / pu[:, 598:603].sum(1)
        r340 = pf[:, 338:343].sum(1) / pu[:, 338:343].sum(1)
        nz = np.where((r600 > 1.3) | (r340 > 1.3))[0]
        print(f"    {tag}: col600 比值行均值 {r600.mean():.3f}，col340 {r340.mean():.3f}；"
              f"强增强行 {nz.min() if len(nz) else '-'}-{nz.max() if len(nz) else '-'} "
              f"（共 {len(nz)}/32 行）")


if __name__ == "__main__":
    main()
