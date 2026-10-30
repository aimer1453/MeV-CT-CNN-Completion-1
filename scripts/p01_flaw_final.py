"""P0-1 前置检查第三步（i）：最终判别 —— 几何差异 vs 材质差异 vs 局部缺陷。

待判：
  A. 零计数截断伪影（-log 的 max(I,1e-10)）是否造成"总衰减差 56.5%"的假象
  B. 两扫描的棒半径/栅距是否一致（几何差异）
  C. 棒(-1,+1) 的强负差分是局部缺陷还是系统差异
  D. recon 与 sinogram 的量程关系（sanity check）
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


def main():
    # 载入原始投影与空场
    pf = {a: np.load(FLAW / f"projections/proj_a{a:06.1f}_z+0.npy").astype(np.float64)
          for a in range(360)}
    pu = {a: np.load(UNIF / f"projections/proj_a{a:03d}_z+0.npy").astype(np.float64)
          for a in range(360)}
    empty = np.load(FLAW / "results" / "empty_proj.npy").astype(np.float64)
    e_row = np.maximum(smooth_empty_proj(empty, 2.0)[16], 1e-10)

    # ---------- A. 零计数与总衰减 ----------
    print("=" * 74)
    print("A. 零计数截断伪影检查")
    print("=" * 74)
    zf = zu = 0
    tf = tu = 0.0
    for a in range(360):
        f, u = pf[a][16], pu[a][16]
        zf += int((f <= 0).sum()); zu += int((u <= 0).sum())
        tf += f.sum(); tu += u.sum()
    print(f"  row16 零计数像素-角: flaw={zf}  unif={zu}（各 432000）")
    # 只在双方都 >0 的像素上算 -log 和
    sf = su = n = 0
    for a in range(360):
        f, u = pf[a][16], pu[a][16]
        m = (f > 0) & (u > 0)
        sf += float(np.sum(-np.log(f[m] / e_row[m])))
        su += float(np.sum(-np.log(u[m] / e_row[m])))
        n += int(m.sum())
    print(f"  双方均有效的像素-角: {n}")
    print(f"  有效域 -log 总和: flaw={sf:.1f} unif={su:.1f} → 差 {(sf-su)/su*100:+.2f}%")
    print(f"  总计数: flaw={tf:.0f} unif={tu:.0f} → 差 {(tf-tu)/tu*100:+.2f}%")
    # 零计数像素的差贡献
    dz = 0
    for a in range(360):
        f, u = pf[a][16], pu[a][16]
        m = (f <= 0) & (u > 0)
        dz += int(m.sum())
        m2 = (f > 0) & (u <= 0)
        dz -= int(m2.sum())
    print(f"  unif 零/flaw 非零 的像素-角: {dz}（这些像素的 -log 差被 1e-10 截断放大）")

    # ---------- D. sinogram/recon 量程 ----------
    print()
    print("=" * 74)
    print("D. sinogram 与 recon 量程对照（sanity）")
    print("=" * 74)
    sino_u = np.stack([-np.log(np.maximum(pu[a][16], 1e-10) / e_row) for a in range(360)])
    recon_u = np.load(OUT / "recon_unif.npy")
    print(f"  sinogram(unif): mean={sino_u.mean():.4f} max={sino_u.max():.4f} "
          f"p99={np.percentile(sino_u, 99):.4f}")
    print(f"  recon(unif):    mean={recon_u.mean():.6f} max={recon_u.max():.6f}")
    print(f"  比值 recon_mean/sino_mean = {recon_u.mean()/sino_u.mean():.4f}")

    # ---------- B. 棒半径与栅距 ----------
    print()
    print("=" * 74)
    print("B. 棒半径/栅距测量（中心棒径向轮廓 50% 穿越点）")
    print("=" * 74)
    for tag, recon in [("unif", recon_u), ("flaw", np.load(OUT / "recon_flaw.npy"))]:
        y, x = np.ogrid[:1200, :1200]
        r = np.sqrt((x - CENTER) ** 2 + (y - CENTER) ** 2)
        ri = r.astype(int).ravel()
        prof = np.bincount(ri, weights=recon.ravel(), minlength=601)
        cnt = np.bincount(ri, minlength=601).astype(float)
        cnt[cnt == 0] = 1
        prof = prof / cnt
        core = prof[10:60].mean()
        half = core * 0.5
        # 从 r=60 向外找第一个跌破 half 的位置
        edge = next((i for i in range(60, 200) if prof[i] < half), None)
        print(f"  {tag}: 中心棒芯 mean={core:.5f}，50% 穿越点 r≈{edge} px "
              f"（= {edge*0.01:.2f} cm，棒芯半径 0.75 cm = 75 px）")
        print(f"      轮廓 r=60..120: " + " ".join(f"{prof[i]:.4f}" for i in range(60, 121, 10)))

    # 栅距：相邻棒心距
    print()
    print("  栅距（相邻棒心距，用轮廓峰/相关法）:")
    for tag, recon in [("unif", recon_u), ("flaw", np.load(OUT / "recon_flaw.npy"))]:
        # 沿中心行 y=600 的剖面，找 3 个棒峰
        row = recon[600, :]
        from scipy.ndimage import gaussian_filter1d
        rs = gaussian_filter1d(row, 5)
        pk = [i for i in range(50, 1150) if rs[i] == max(rs[i - 40:i + 41]) and rs[i] > 0.5 * rs.max()]
        if len(pk) >= 3:
            print(f"    {tag}: 峰列 {pk} → 栅距 {pk[1]-pk[0]} / {pk[2]-pk[1]} px")
        else:
            print(f"    {tag}: 峰列 {pk}")

    # ---------- C. 棒(-1,+1) 特征 ----------
    print()
    print("=" * 74)
    print("C. 棒(-1,+1) 特征的角度签名（正弦图上该缺陷的轨迹）")
    print("=" * 74)
    # 该棒在 (600-260, 600-260) = (340,340)。缺陷若在此棒内，正弦图差分的负迹
    # 应围绕 col = 600 ± 410*sin/cos... 先用简单法：每角度找 row16 差分最小列，
    # 只看差分 < -0.3 的强信号
    from scipy.ndimage import gaussian_filter1d
    trace = []
    for a in range(360):
        d = gaussian_filter1d(-np.log(np.maximum(pf[a][16], 1e-10) / e_row)
                              - -np.log(np.maximum(pu[a][16], 1e-10) / e_row), 3)
        j = int(np.argmin(d))
        trace.append((j, float(d[j])))
    cols = np.array([t[0] for t in trace])
    vals = np.array([t[1] for t in trace])
    strong = vals < -0.3
    print(f"  强信号(< -0.3)角度数: {strong.sum()}/360")
    if strong.sum():
        print(f"  这些角度的差分最小列: 范围 {cols[strong].min()}-{cols[strong].max()}，"
              f"中位 {int(np.median(cols[strong]))}")
    # 该棒中心在图像 (340,340)；平行束下其投影列 ≈ 600 - 340 = 260（无放大）
    # 有放大则为 600 - 410 = 190。打印两个候选
    print(f"  候选：无放大棒心投影列 ≈ 600-340 = 260；带放大 ≈ 600-410 = 190")
    print(f"  全角度差分最小列直方图(12 bin, 0-1200): "
          f"{np.histogram(cols, bins=12, range=(0,1200))[0].tolist()}")


if __name__ == "__main__":
    main()
