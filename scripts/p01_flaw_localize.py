"""P0-1 前置检查第三步（c）：重建定位缺陷的 (i, j) 位置与类型。

做法：缺陷数据与均匀 3.0MeV 走同一条管线（smooth_empty(2.0) + row16 +
ring_correction(fw,5) + FBP hamming），差分图即缺陷的像。
另用正弦图轨迹拟合作独立交叉验证。

⚠️ 必须有 `if __name__ == "__main__"` 保护：`tomopy.remove_stripe_fw` 内部用
multiprocessing（`mproc.distribute_jobs` → `mp.Manager()`），Windows spawn 模式下
子进程会重入主脚本，没有保护会无限递归（实测 RuntimeError + 进程风暴）。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "3_3_geant4"))
sys.path.insert(0, str(ROOT / "scripts"))

from quick_reconstruct import (compute_sinogram_smooth_empty, ring_correction,  # noqa: E402
                               reconstruct_fbp)

FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"
OUT = ROOT / "tmp_p01_flaw"

angles = list(range(360))
PITCH_CM = 2.6          # 棒栅距
PX_CM = 0.01            # 探测器像素尺寸 cm
CENTER = 600            # 图像中心像素


def build_recon(scan_dir, naming):
    proj_dir = scan_dir / "projections"
    if naming == "float":
        proj = {a: np.load(proj_dir / f"proj_a{a:06.1f}_z+0.npy").astype(np.float64)
                for a in angles}
    else:
        proj = {a: np.load(proj_dir / f"proj_a{a:03d}_z+0.npy").astype(np.float64)
                for a in angles}
    empty = np.load(scan_dir / "results" / "empty_proj.npy").astype(np.float64)
    sino = compute_sinogram_smooth_empty(proj, empty, 16, angles, sigma=2.0)
    sino = ring_correction(sino, method='fw', level=5)
    recon = reconstruct_fbp(sino, angles)
    return recon, sino, proj, empty


def main():
    OUT.mkdir(exist_ok=True)
    print("重建缺陷数据（360 角 dense FBP，row16）...", flush=True)
    recon_f, sino_f, proj_f, empty_f = build_recon(FLAW, "float")
    print("重建均匀 3.0MeV（同管线）...", flush=True)
    recon_u, sino_u, proj_u, empty_u = build_recon(UNIF, "int")

    np.save(OUT / "recon_flaw.npy", recon_f)
    np.save(OUT / "recon_unif.npy", recon_u)
    np.save(OUT / "diff.npy", recon_f - recon_u)

    print()
    print("=" * 72)
    print("缺陷定位（重建差分图）")
    print("=" * 72)
    diff = recon_f - recon_u
    print(f"recon_flaw 范围 [{recon_f.min():.5f}, {recon_f.max():.5f}]；"
          f"recon_unif 范围 [{recon_u.min():.5f}, {recon_u.max():.5f}]")
    print(f"diff 范围 [{diff.min():.5f}, {diff.max():.5f}]；mean={diff.mean():.3e}")

    i = np.unravel_index(np.argmin(diff), diff.shape)
    print(f"diff 最小值: row={i[0]}, col={i[1]}, 值={diff[i]:.5f}")
    print(f"  → 物理位置: x={(i[1]-CENTER)*PX_CM:+.2f} cm, y={(CENTER-i[0])*PX_CM:+.2f} cm（图像中心为原点）")
    imax = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"diff 最大值: row={imax[0]}, col={imax[1]}, 值={diff[imax]:.5f} "
          f"（x={(imax[1]-CENTER)*PX_CM:+.2f}, y={(CENTER-imax[0])*PX_CM:+.2f} cm）")

    thr = diff.min() * 0.5
    mask = diff < thr
    print(f"diff < 50%最小值 的像素数: {mask.sum()}")
    if mask.sum():
        rs, cs = np.where(mask)
        print(f"  行范围 {rs.min()}-{rs.max()}，列范围 {cs.min()}-{cs.max()}")
        print(f"  质心: row={rs.mean():.1f}, col={cs.mean():.1f} "
              f"→ x={(cs.mean()-CENTER)*PX_CM:+.2f} cm, y={(CENTER-rs.mean())*PX_CM:+.2f} cm")
        gx = (cs.mean() - CENTER) * PX_CM / PITCH_CM
        gy = (CENTER - rs.mean()) * PX_CM / PITCH_CM
        print(f"  棒阵坐标（单位=栅距）: ({gx:+.2f}, {gy:+.2f})；最近整数棒位: "
              f"({round(gx)}, {round(gy)})")

    print()
    print("9 根棒中心（图像像素 / 物理 cm）:")
    for bi in (-1, 0, 1):
        for bj in (-1, 0, 1):
            r = CENTER - bj * PITCH_CM / PX_CM
            c = CENTER + bi * PITCH_CM / PX_CM
            print(f"  棒({bi:+d},{bj:+d}): row={r:.0f} col={c:.0f} → "
                  f"x={bi*PITCH_CM:+.1f} y={bj*PITCH_CM:+.1f} cm")

    print()
    print("=" * 72)
    print("交叉验证：正弦图上缺陷轨迹（每角度最大正差位置）")
    print("=" * 72)
    cols = []
    for a in range(360):
        d = proj_f[a][16] - proj_u[a][16]
        cols.append(int(np.argmax(d)))
    cols = np.array(cols)
    print(f"轨迹列范围: {cols.min()}-{cols.max()}（探测器 1200 列，中心 600）")
    t = np.deg2rad(np.arange(360))
    A = np.c_[np.ones(360), np.cos(t), np.sin(t)]
    coef, *_ = np.linalg.lstsq(A, cols, rcond=None)
    c0, ac, as_ = coef
    amp = np.hypot(ac, as_)
    phi = np.arctan2(-as_, ac)
    pred = A @ coef
    rms = float(np.sqrt(np.mean((cols - pred) ** 2)))
    print(f"拟合 col = {c0:.1f} + {amp:.1f}*cos(t+{np.rad2deg(phi):.1f}°)，RMS 残差 {rms:.1f} px")
    M = 1 + 31.7 / 55.0
    print(f"几何放大率 M = 1 + 31.7/55 = {M:.3f}")
    print(f"⇒ 缺陷距旋转中心约 {amp / M * PX_CM:.2f} cm（{amp / M * PX_CM / PITCH_CM:.2f} 个栅距）")
    x0 = amp / M * PX_CM * np.cos(phi)
    y0 = amp / M * PX_CM * np.sin(phi)
    print(f"⇒ 缺陷物体坐标约 x={x0:+.2f} cm, y={y0:+.2f} cm（拟合相位约定，象限需结合重建图确认）")

    print()
    print("=" * 72)
    print("空场扫描对比（flaw empty vs 均匀 empty，row16）")
    print("=" * 72)
    e_f = empty_f[16]
    e_u = empty_u[16]
    print(f"row16 均值: flaw={e_f.mean():.4f} unif={e_u.mean():.4f}")
    print(f"max|diff| = {np.abs(e_f - e_u).max():.4f}；mean|diff| = {np.abs(e_f - e_u).mean():.4f}")
    print(f"相关系数 = {np.corrcoef(e_f, e_u)[0, 1]:.6f}")


if __name__ == "__main__":
    main()
