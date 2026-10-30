"""P0-1 前置检查第三步（e）：差分模式可视化 + ASCII 粗图 + 与历史几何版本的关系。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "3_3_geant4"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "tmp_p01_flaw"
FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"

CENTER = 600


def main():
    recon_f = np.load(OUT / "recon_flaw.npy")
    recon_u = np.load(OUT / "recon_unif.npy")
    diff = recon_f - recon_u

    # ASCII 粗图（1200 -> 60 格，每格 20 px 取均值）
    print("diff 的 ASCII 图（60x60，每格 20x20 px 均值；'#' 强负，'+' 负，'.' 近零，'o' 正）:")
    g = diff.reshape(60, 20, 60, 20).mean(axis=(1, 3))
    lo, hi = np.percentile(g, [2, 98])
    chars = " .:-=+*#%@"
    for row in range(60):
        line = ""
        for col in range(60):
            v = (g[row, col] - lo) / (hi - lo + 1e-12)
            k = int(np.clip(v * (len(chars) - 1), 0, len(chars) - 1))
            line += chars[k]
        print("   " + line)
    print(f"   （色窗 [{lo:.5f}, {hi:.5f}]；行=图像行（上=y+），列=图像列（右=x+）；"
          f"中心格 (30,30)；四角棒位格 (10,10),(10,50),(50,10),(50,50)）")

    # 正弦图差分 ASCII（360x1200 -> 36x60）
    sino_diff = np.zeros((360, 1200))
    from quick_reconstruct import smooth_empty_proj
    empty = np.load(FLAW / "results" / "empty_proj.npy").astype(np.float64)
    e_row = np.maximum(smooth_empty_proj(empty, 2.0)[16], 1e-10)
    proj_f = {a: np.load(FLAW / f"projections/proj_a{a:06.1f}_z+0.npy").astype(np.float64)
              for a in range(360)}
    proj_u = {a: np.load(UNIF / f"projections/proj_a{a:03d}_z+0.npy").astype(np.float64)
              for a in range(360)}
    for a in range(360):
        lf = -np.log(np.maximum(proj_f[a][16], 1e-10) / e_row)
        lu = -np.log(np.maximum(proj_u[a][16], 1e-10) / e_row)
        sino_diff[a] = lf - lu
    print()
    print("正弦图差分 ASCII（36x60，每格 10x20；'#' 强负 = 缺陷迹）:")
    g2 = sino_diff.reshape(36, 10, 60, 20).mean(axis=(1, 3))
    lo2, hi2 = np.percentile(g2, [1, 99])
    for row in range(36):
        line = ""
        for col in range(60):
            v = (g2[row, col] - lo2) / (hi2 - lo2 + 1e-12)
            k = int(np.clip(v * (len(chars) - 1), 0, len(chars) - 1))
            line += chars[k]
        print(f"   a={row*10:3d} {line}")
    print(f"   （色窗 [{lo2:.5f}, {hi2:.5f}]；列=探测器列（中心 30 格=600 列））")

    # 出图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    p1, p99 = np.percentile(recon_u, [0.5, 99.5])
    for ax, img, title in zip(
            axes,
            [recon_u, recon_f, diff],
            ["Uniform 3.0 MeV (Jul 26 scan)", "Flaw scan (Aug 4 scan)",
             "Difference (flaw - uniform)"]):
        if title.startswith("Difference"):
            v = np.percentile(np.abs(diff), 99.5)
            im = ax.imshow(img, cmap="RdBu_r", vmin=-v, vmax=v)
            fig.colorbar(im, ax=ax, fraction=0.046)
        else:
            im = ax.imshow(img, cmap="gray", vmin=p1, vmax=p99)
            fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (pixel)"); ax.set_ylabel("y (pixel)")
        # 标 9 根棒位
        for bi in (-1, 0, 1):
            for bj in (-1, 0, 1):
                ax.plot(CENTER + bi * 260, CENTER - bj * 260, "c+", ms=6, mew=1)
    fig.suptitle("P0-1 flaw data check: dense 360° FBP (smooth_empty 2.0 + ring fw5, row16)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "flaw_vs_uniform_recon.png", dpi=150)
    print()
    print(f"图已存: {OUT / 'flaw_vs_uniform_recon.png'}")


if __name__ == "__main__":
    main()
