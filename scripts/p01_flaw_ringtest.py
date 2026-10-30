"""P0-1 前置检查第三步（k）：判别径向振荡模式的来源。

假设：差分图的「蓝芯+红环+蓝外环」是 ring_correction(fw,5) 对两正弦图施加不同
角度向平滑造成的模糊度差异，而非材质差异。
测试：不做 ring_correction 重建两扫描，重新差分，看模式是否消失。
附带：iradon 量程测试（物理单位核对）。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "3_3_geant4"))
sys.path.insert(0, str(ROOT / "scripts"))

from quick_reconstruct import compute_sinogram_smooth_empty, reconstruct_fbp  # noqa: E402

OUT = ROOT / "tmp_p01_flaw"
FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"
CENTER = 600


def main():
    angles = list(range(360))

    # ---- 附带测试：iradon 量程 ----
    from skimage.transform import iradon
    sino_test = np.ones((180, 200)) * 1.0     # 常数正弦图
    rec = iradon(sino_test.T, theta=np.arange(180), filter_name='hamming')
    print(f"[iradon 量程] 常数 sinogram=1.0（180 角）→ recon 中心值 {rec[100,100]:.6f}")
    sino_test2 = np.ones((360, 200)) * 1.0
    rec2 = iradon(sino_test2.T, theta=np.arange(360), filter_name='hamming')
    print(f"[iradon 量程] 常数 sinogram=1.0（360 角）→ recon 中心值 {rec2[100,100]:.6f}")
    print("  （若远小于 1，说明管线 recon 值比物理线积分小 1/(2N) 量级；"
          "相对比较不受影响）")

    # ---- 无 ring_correction 重建 ----
    pf = {a: np.load(FLAW / f"projections/proj_a{a:06.1f}_z+0.npy").astype(np.float64)
          for a in angles}
    pu = {a: np.load(UNIF / f"projections/proj_a{a:03d}_z+0.npy").astype(np.float64)
          for a in angles}
    ef = np.load(FLAW / "results" / "empty_proj.npy").astype(np.float64)
    eu = np.load(UNIF / "results" / "empty_proj.npy").astype(np.float64)

    print()
    print("无 ring_correction 重建中...")
    sino_f = compute_sinogram_smooth_empty(pf, ef, 16, angles, sigma=2.0)
    sino_u = compute_sinogram_smooth_empty(pu, eu, 16, angles, sigma=2.0)
    rec_f = reconstruct_fbp(sino_f, angles)
    rec_u = reconstruct_fbp(sino_u, angles)
    d = rec_f - rec_u
    np.save(OUT / "recon_noring_flaw.npy", rec_f)
    np.save(OUT / "recon_noring_unif.npy", rec_u)
    np.save(OUT / "diff_noring.npy", d)
    print(f"  无环校正 diff: 范围 [{d.min():.6f}, {d.max():.6f}]")

    # 径向轮廓对比（中心棒）
    y, x = np.ogrid[:1200, :1200]
    r = np.sqrt((x - CENTER) ** 2 + (y - CENTER) ** 2).astype(int)

    def radial(img):
        p = np.bincount(r.ravel(), weights=img.ravel(), minlength=601)
        c = np.bincount(r.ravel(), minlength=601).astype(float)
        c[c == 0] = 1
        return p / c

    print()
    print("中心棒径向差分轮廓（无环校正 vs 有环校正）:")
    p_nr = radial(d)
    p_r = radial(np.load(OUT / "diff.npy"))
    print(f"  {'r':>4} {'无环校正':>12} {'有环校正':>12}")
    for rr in range(0, 161, 20):
        print(f"  {rr:>4} {p_nr[rr]:>12.6f} {p_r[rr]:>12.6f}")

    # 各棒芯与环（无环校正）
    def disk(cx, cy, rad):
        return (x - cx) ** 2 + (y - cy) ** 2 <= rad ** 2
    print()
    print("无环校正的逐棒差分（芯 r<=60 / 环 75-95）:")
    for bi in (-1, 0, 1):
        for bj in (-1, 0, 1):
            c, rr_ = CENTER + bi * 260, CENTER - bj * 260
            core, ring = disk(c, rr_, 60), disk(c, rr_, 95) & ~disk(c, rr_, 75)
            print(f"  棒({bi:+d},{bj:+d}): Δ芯={d[core].mean():+.6f} Δ环={d[ring].mean():+.6f}")

    # 有环校正版对照（同位置）
    d_r = np.load(OUT / "diff.npy")
    print()
    print("有环校正的逐棒差分（芯 r<=60 / 环 75-95）:")
    for bi in (-1, 0, 1):
        for bj in (-1, 0, 1):
            c, rr_ = CENTER + bi * 260, CENTER - bj * 260
            core, ring = disk(c, rr_, 60), disk(c, rr_, 95) & ~disk(c, rr_, 75)
            print(f"  棒({bi:+d},{bj:+d}): Δ芯={d_r[core].mean():+.6f} Δ环={d_r[ring].mean():+.6f}")


if __name__ == "__main__":
    main()
