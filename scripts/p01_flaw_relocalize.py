"""P0-1 第一步/第二步：缺陷重新定位 + 轴映射标定。

已验证：flaw 角度均匀 1°（0..359）、宏与均匀 3.0MeV 逐字一致。
本脚本回答：
1. 圆盘之谜：recon_f - recon_u（分别 ring_correction 后 FBP）为何是团块不是环？
   对比四条路径：
   a) 分别 smooth+ring+FBP 后相减（p01_flaw_localize.py 原路径）
   b) 分别 smooth+FBP（无 ring）后相减
   c) 差分 sino（smooth 后相减）直接 FBP
   d) raw -log 差分 sino 直接 FBP（无 smooth）
   FBP 线性 → b/c 应一致且为环；若 c/d 是环而 a 是团块 → ring_correction 非线性伪影
2. 缺陷棒 (i,j) 定位：raw -log 差分 sino 的 FBP 像（环 vs 团块、半径、质心、相位）
3. 90° 差分法：D_a = proj_f[a] - proj_f[a+90] 的正弦轨迹拟合（amp, phi）
4. 轴映射标定：图像 (row,col) -> 物理 (x,z) 方向关系

必须有 __main__ 保护（tomopy ring_correction multiprocessing）。
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "3_3_geant4"))
sys.path.insert(0, str(ROOT / "scripts"))

from quick_reconstruct import compute_sinogram_smooth_empty, ring_correction, reconstruct_fbp  # noqa: E402

FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"
OUT = ROOT / "tmp_p01_flaw"

angles = list(range(360))
CENTER = 600
PITCH_CM = 2.6
PX_CM = 0.01


def load_all():
    proj_f = {a: np.load(FLAW / "projections" / f"proj_a{a:06.1f}_z+0.npy").astype(np.float64)
              for a in angles}
    proj_u = {a: np.load(UNIF / "projections" / f"proj_a{a:03d}_z+0.npy").astype(np.float64)
              for a in angles}
    empty_f = np.load(FLAW / "results" / "empty_proj.npy").astype(np.float64)
    empty_u = np.load(UNIF / "results" / "empty_proj.npy").astype(np.float64)
    return proj_f, proj_u, empty_f, empty_u


def analyze_blob(name, img, diff_min_frac=0.5):
    """对差分图做连通域分析。"""
    print(f"  [{name}] range [{img.min():.5f}, {img.max():.5f}] mean {img.mean():.3e}")
    i = np.unravel_index(np.argmin(img), img.shape)
    print(f"    最小值: row={i[0]} col={i[1]} val={img[i]:.5f}")
    thr = img.min() * diff_min_frac
    mask = img < thr
    lab, n = ndimage.label(mask)
    if n == 0:
        print("    无连通域")
        return
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    for k in order[:3]:
        m = lab == (k + 1)
        rs, cs = np.where(m)
        cy, cx = rs.mean(), cs.mean()
        r_eq = np.sqrt(sizes[k] / np.pi)
        r_center = np.hypot(cx - CENTER, cy - CENTER)
        print(f"    连通域{k+1}: {int(sizes[k])} px, 质心 (row={cy:.0f}, col={cx:.0f})"
              f" → x={(cx-CENTER)*PX_CM:+.2f} z={(CENTER-cy)*PX_CM:+.2f} cm, "
              f"等效半径 {r_eq:.0f} px, 距图像中心 {r_center:.0f} px")
    # 径向剖面（以中心为原点的环强度）
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    rr = np.hypot(xx - CENTER, yy - CENTER).astype(int)
    prof = {}
    for R in [260, 300, 340, 368, 400, 440, 500]:
        sel = (rr == R)
        if sel.sum():
            prof[R] = float(img[sel].mean())
    print("    径向均值: " + "  ".join(f"R={R}:{v:.5f}" for R, v in prof.items()))


def sine_fit(proj_f, proj_u, mode):
    """正弦图域缺陷轨迹拟合。

    mode='argmax': D_a = proj_f[a]-proj_u[a] 每角度 argmax 列
    mode='flaw90': D_a = proj_f[a]-proj_f[a+90]（flaw 自身 90° 差分，抵消 D4 对称的包壳环）
    """
    cols, wts = [], []
    for a in angles:
        if mode == "argmax":
            d = proj_f[a][16] - proj_u[a][16]
        else:
            d = proj_f[a][16] - proj_f[(a + 90) % 360][16]
        c = int(np.argmax(d))
        cols.append(c)
        wts.append(float(d[c]))
    cols = np.array(cols, dtype=float)
    t = np.deg2rad(np.array(angles, dtype=float))
    A = np.c_[np.ones(360), np.cos(t), np.sin(t)]
    coef, *_ = np.linalg.lstsq(A, cols, rcond=None)
    c0, Ac, Bs = coef
    amp = np.hypot(Ac, Bs)
    phi = np.degrees(np.arctan2(-Bs, Ac)) % 360
    pred = A @ coef
    rms = float(np.sqrt(np.mean((cols - pred) ** 2)))
    print(f"  [{mode}] col = {c0:.1f} + {amp:.1f}·cos(θ+{phi:.1f}°)，RMS 残差 {rms:.1f} px")
    r_cm = amp * PX_CM  # M_eff=1（准直器正交投影）
    print(f"    → 缺陷距中心 {r_cm:.2f} cm = {r_cm/PITCH_CM:.2f} 个栅距；"
          f"初始方向 (x0,y0) = ({r_cm*np.cos(np.radians(phi)):+.2f}, "
          f"{r_cm*np.sin(np.radians(phi)):+.2f}) cm")
    # 象限解释（棒位取整到 -1/0/1 栅距）
    gi, gj = r_cm * np.cos(np.radians(phi)) / PITCH_CM, r_cm * np.sin(np.radians(phi)) / PITCH_CM
    print(f"    → 棒阵坐标 ({gi:+.2f}, {gj:+.2f}) → 最近整数 ({round(gi)}, {round(gj)})")
    # 拟合残差大的角度（可能被干扰的角度）
    resid = np.abs(cols - pred)
    bad = np.argsort(resid)[::-1][:5]
    print(f"    残差最大角度: {[(int(a), float(resid[a])) for a in bad]}")


def main():
    OUT.mkdir(exist_ok=True)
    print("加载 720 个投影...", flush=True)
    proj_f, proj_u, empty_f, empty_u = load_all()
    print("empty_proj 逐位相同:", np.array_equal(empty_f, empty_u),
          " max|diff| =", float(np.abs(empty_f - empty_u).max()))

    # ---- 正弦图（row16, sigma=2.0 smooth_empty）----
    print("\n构建 smooth_empty 正弦图...", flush=True)
    sino_f = compute_sinogram_smooth_empty(proj_f, empty_f, 16, angles, sigma=2.0)
    sino_u = compute_sinogram_smooth_empty(proj_u, empty_u, 16, angles, sigma=2.0)

    # ---- 路径 a: 原脚本路径（ring 后分别 FBP 再相减）----
    print("\n[路径 a] smooth + ring_correction(fw,5) 后分别 FBP 再相减（原脚本路径）", flush=True)
    rf = reconstruct_fbp(ring_correction(sino_f, method='fw', level=5), angles)
    ru = reconstruct_fbp(ring_correction(sino_u, method='fw', level=5), angles)
    diff_a = rf - ru
    analyze_blob("a: ring后差分", diff_a)

    # ---- 路径 b: 无 ring 分别 FBP 再相减 ----
    print("\n[路径 b] smooth 后分别 FBP（无 ring）再相减", flush=True)
    rf2 = reconstruct_fbp(sino_f, angles)
    ru2 = reconstruct_fbp(sino_u, angles)
    diff_b = rf2 - ru2
    analyze_blob("b: 无ring差分", diff_b)

    # ---- 路径 c: 差分 sino 直接 FBP ----
    print("\n[路径 c] 差分 sino（smooth 后相减）直接 FBP", flush=True)
    diff_c = reconstruct_fbp(sino_f - sino_u, angles)
    analyze_blob("c: 差分sino FBP", diff_c)

    # ---- 路径 d: raw -log 差分 sino 直接 FBP（无 smooth）----
    print("\n[路径 d] raw -log 差分 sino 直接 FBP（无 smooth 无 ring）", flush=True)
    d_sino = np.zeros((360, 1200))
    for a in angles:
        d_sino[a] = -np.log(proj_f[a][16] / empty_f[16] + 1e-12) \
                    - (-np.log(proj_u[a][16] / empty_u[16] + 1e-12))
    diff_d = reconstruct_fbp(d_sino, angles)
    analyze_blob("d: raw差分 FBP", diff_d)

    np.save(OUT / "diff_d_raw.npy", diff_d)
    np.save(OUT / "diff_c_smooth.npy", diff_c)

    # ---- 正弦轨迹拟合 ----
    print("\n正弦图域缺陷轨迹拟合（两种模式）")
    sine_fit(proj_f, proj_u, "argmax")
    sine_fit(proj_f, proj_u, "flaw90")

    # ---- 轴映射标定 ----
    print("\n" + "=" * 72)
    print("轴映射标定（几何代码推导）")
    print("=" * 72)
    print("  传感器放置无旋转: posSensor=(sensorX, sensorY, sensorZ)，sensorX=(i+0.5)*0.01cm")
    print("  copyNo=(j+16)*1200+(i+600) → row=j+16 ↔ Z，col=i+600 ↔ X")
    print("  → 数组 col 增大 = 物理 X 增大（同向）")
    print("  → 数组 row 增大 = 物理 Z 增大（同向）；但 imshow 把 row0 放顶部")
    print("    ⇒ 屏幕上「上」= row 小 = 物理 Z 小（下）；图像显示 y 轴相对物理 Z 翻转")
    print("  组件绕 +Z 右手旋转 rotateZ(θ): (x,y) → (x cosθ − y sinθ, x sinθ + y cosθ)")
    print("  源 (0,+55,0) 方向 (0,−1,0)，准直器 ±0.16° 接受角 → 近垂直采样")
    print("  物点 (x,y) → 投影列 col = 600 + x/0.01（X 不翻转，M_eff≈1）")
    print("  θ=0 时组件局部 = 世界；棒 (i,j) 局部坐标 (i·2.6, j·2.6) cm")


if __name__ == "__main__":
    main()
