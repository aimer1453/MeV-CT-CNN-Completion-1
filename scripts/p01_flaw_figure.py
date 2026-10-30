"""P0-1 前置检查第三步（j）：生成 flaw vs uniform 对比图（修正 import bug）。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "3_3_geant4"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "tmp_p01_flaw"
CENTER = 600


def main():
    recon_f = np.load(OUT / "recon_flaw.npy")
    recon_u = np.load(OUT / "recon_unif.npy")
    diff = recon_f - recon_u
    recon_d = np.load(OUT / "recon_defect_only.npy")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    p1, p99 = np.percentile(recon_u, [0.5, 99.5])

    ax = axes[0, 0]
    im = ax.imshow(recon_u, cmap="gray", vmin=p1, vmax=p99)
    ax.set_title("Uniform 3.0 MeV (3_3/, Jul 26 scan)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im = ax.imshow(recon_f, cmap="gray", vmin=p1, vmax=p99)
    ax.set_title("Flaw scan (3_3_flaw/, Aug 4 scan)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    v = np.percentile(np.abs(diff), 99.5)
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title("Difference (flaw - uniform)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    v2 = np.percentile(np.abs(recon_d), 99.5)
    im = ax.imshow(recon_d, cmap="RdBu_r", vmin=-v2, vmax=v2)
    ax.set_title("Defect-only image (sinogram-diff reconstruction)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes.ravel():
        for bi in (-1, 0, 1):
            for bj in (-1, 0, 1):
                ax.plot(CENTER + bi * 260, CENTER - bj * 260, "c+", ms=7, mew=1)
        ax.set_xlabel("x (pixel)"); ax.set_ylabel("y (pixel)")

    fig.suptitle("P0-1 flaw-data check: dense 360-deg FBP, smooth_empty(2.0) + ring fw5, row 16\n"
                 "cyan + = nominal rod centres (pitch 2.6 cm = 260 px); rods sit at +/-260 px, NOT +/-410 px",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "flaw_vs_uniform_recon.png", dpi=150)
    print(f"saved: {OUT / 'flaw_vs_uniform_recon.png'}")

    # 差分图局部放大：(-1,+1) 角棒
    fig2, ax = plt.subplots(1, 2, figsize=(11, 5))
    v = np.percentile(np.abs(diff), 99.5)
    ax[0].imshow(diff[200:500, 200:500], cmap="RdBu_r", vmin=-v, vmax=v)
    ax[0].set_title("diff zoom: rod(-1,+1) corner, img[200:500,200:500]", fontsize=10)
    ax[1].imshow(diff[450:750, 450:750], cmap="RdBu_r", vmin=-v, vmax=v)
    ax[1].set_title("diff zoom: centre rod, img[450:750,450:750]", fontsize=10)
    for a in ax:
        a.set_xlabel("x (pixel)"); a.set_ylabel("y (pixel)")
    fig2.tight_layout()
    fig2.savefig(OUT / "flaw_diff_zoom.png", dpi=150)
    print(f"saved: {OUT / 'flaw_diff_zoom.png'}")


if __name__ == "__main__":
    main()
