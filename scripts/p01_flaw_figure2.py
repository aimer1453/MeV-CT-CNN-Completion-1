"""P0-1 修正版出图：缺陷重新定位 + 轴映射标定证据图。

四联：
(a) 去直流差分 recon 全图（替代 ring 伪影版）
(b) 中心放大 + R=368 环 + 9 棒格点（标记缺陷角棒/中心条纹）
(c) R=368 环角向剖面（图像极角 135° 峰 vs 真实缺陷方向 225° → -90° 旋转）
(d) 360 角累加 row16 列比值（col 340 缺陷棒 / col 600 中心条纹 / col 860 正常）
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tmp_p01_flaw"
FIGDIR = ROOT / "docs" / "figs" / "p01_flaw_check"
CENTER = 600


def build_col_ratio():
    """360 角累加 row16 列比值（flaw/uniform）。"""
    import os
    cache = OUT / "col_ratio.npy"
    if cache.exists():
        return np.load(cache)
    FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
    UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"
    sf = np.zeros(1200)
    su = np.zeros(1200)
    for a in range(360):
        sf += np.load(FLAW / "projections" / f"proj_a{a:06.1f}_z+0.npy").astype(np.float64)[16]
        su += np.load(UNIF / "projections" / f"proj_a{a:03d}_z+0.npy").astype(np.float64)[16]
    r = sf / su
    OUT.mkdir(exist_ok=True)
    np.save(cache, r)
    print(f"col ratio: col340={r[340]:.4f} col600={r[600]:.4f} col860={r[860]:.4f}")
    return r


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.load(OUT / "diff_dc.npy")          # 去直流差分 recon
    r_col = build_col_ratio()                    # 360 角累加 row16 列比值

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # (a) 全图
    ax = axes[0, 0]
    v = np.percentile(np.abs(img), 99.5)
    im = ax.imshow(img, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.add_patch(plt.Circle((CENTER, CENTER), 368, fill=False, color="k", lw=0.8, ls="--"))
    ax.add_patch(plt.Circle((CENTER, CENTER), 260, fill=False, color="gray", lw=0.6, ls=":"))
    ax.set_title("(a) flaw - uniform, DC-removed FBP\n"
                 "(ring_correction removed; centre spike = column-stripe artefact)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (b) 中心放大
    ax = axes[0, 1]
    sub = img[450:751, 450:751]
    im = ax.imshow(sub, cmap="RdBu_r", vmin=-v, vmax=v, extent=[450, 751, 751, 450])
    for bi in (-1, 0, 1):
        for bj in (-1, 0, 1):
            c, r0 = CENTER + bi * 260, CENTER - bj * 260
            ax.plot(c, r0, "+", color="lime", ms=8, mew=1.2)
    ax.plot(345, 345, "o", mfc="none", mec="yellow", ms=14, mew=2)
    ax.annotate("flawed rod\n(true (-1,-1) -> image 135 deg)", (345, 345), (400, 480),
                color="yellow", fontsize=8, arrowprops=dict(arrowstyle="->", color="yellow"))
    ax.plot(600, 600, "s", mfc="none", mec="cyan", ms=12, mew=2)
    ax.annotate("centre column stripe\n(rotation-axis feature)", (600, 600), (480, 560),
                color="cyan", fontsize=8, arrowprops=dict(arrowstyle="->", color="cyan"))
    ax.set_title("(b) centre zoom, 9-rod grid (lime +); dashed R=368 ring", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (c) 环角向剖面
    ax = axes[1, 0]
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    rr = np.hypot(xx - CENTER, yy - CENTER)
    ang = np.degrees(np.arctan2(-(yy - CENTER), xx - CENTER)) % 360
    prof = np.full(36, np.nan)
    for k in range(36):
        sel = (np.abs(rr - 368) <= 2) & (ang >= k * 10) & (ang < (k + 1) * 10)
        if sel.sum():
            prof[k] = img[sel].mean()
    xs = np.arange(36) * 10 + 5
    ax.bar(xs, prof, width=9, color="steelblue")
    ax.axvline(135, color="red", ls="--", lw=1)
    ax.axvline(225, color="green", ls="--", lw=1)
    ax.text(137, ax.get_ylim()[0] * 0.9, "image peak 135 deg", color="red", fontsize=8)
    ax.text(190, ax.get_ylim()[0] * 0.9, "true defect 225 deg\n(world coords)", color="green", fontsize=8)
    ax.set_xlabel("image polar angle (deg)")
    ax.set_title("(c) R=368 ring profile: image angle = world angle - 90 deg", fontsize=9)

    # (d) 列比值
    ax = axes[1, 1]
    ax.plot(np.arange(1200), r_col, lw=0.8)
    for c, lbl, col in [(340, "flawed rod col", "red"), (600, "centre col", "cyan"),
                        (860, "normal rods col", "green")]:
        ax.axvline(c, color=col, ls="--", lw=1)
        ax.text(c + 5, 1.98, lbl, color=col, fontsize=8, rotation=90, va="top")
    ax.set_xlabel("detector column")
    ax.set_ylabel("flaw/uniform count ratio\n(360 angles summed, row 16)")
    ax.set_ylim(0.95, 2.05)
    ax.set_title("(d) column ratio, 360-angle sum: col 340 = 1.36, col 600 = 1.49,\n"
                 "col 860 = 1.26 (single angle a000: 1.99 / 1.95 / 1.08)", fontsize=9)

    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "flaw_relocalize.png", dpi=150)
    print("saved:", FIGDIR / "flaw_relocalize.png")


if __name__ == "__main__":
    main()
