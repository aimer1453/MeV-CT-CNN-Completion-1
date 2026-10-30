"""P0-1 前置检查第三步（h）：直接看 recon_unif 本体（ASCII + 与既有产物对照）。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tmp_p01_flaw"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"
FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"


def main():
    recon_u = np.load(OUT / "recon_unif.npy")
    recon_f = np.load(OUT / "recon_flaw.npy")
    print(f"recon_unif: min={recon_u.min():.5f} max={recon_u.max():.5f} mean={recon_u.mean():.5f}")
    print(f"recon_flaw: min={recon_f.min():.5f} max={recon_f.max():.5f} mean={recon_f.mean():.5f}")

    # 与既有管线产物对照
    print()
    print("既有管线产物（results/postprocessing）:")
    for p in sorted((UNIF / "results" / "postprocessing").glob("*row16*")):
        print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")
    for name in ["recon_smooth_empty_sigma2.0_sirt_ringfw_l5_row16.npy",
                 "recon_smooth_empty_sigma2.0_fbp_ringfw_l5_row16.npy",
                 "recon_original_fbp_row16.npy", "recon_fbp_row16.npy"]:
        p = UNIF / "results" / "postprocessing" / name
        if p.exists():
            a = np.load(p)
            print(f"  {name}: shape={a.shape} min={a.min():.5f} max={a.max():.5f} mean={a.mean():.5f}")

    # ASCII：recon_unif 本体（60x60）
    print()
    print("recon_unif ASCII（60x60，' '最低 '@'最高）:")
    g = recon_u.reshape(60, 20, 60, 20).mean(axis=(1, 3))
    lo, hi = g.min(), g.max()
    chars = " .:-=+*#%@"
    for row in range(60):
        line = ""
        for col in range(60):
            v = (g[row, col] - lo) / (hi - lo + 1e-12)
            k = int(np.clip(v * (len(chars) - 1), 0, len(chars) - 1))
            line += chars[k]
        print("   " + line)
    print(f"   （范围 [{lo:.5f}, {hi:.5f}]；格 (30,30)=图像中心）")

    # ASCII：recon_flaw 本体
    print()
    print("recon_flaw ASCII（同样式）:")
    g2 = recon_f.reshape(60, 20, 60, 20).mean(axis=(1, 3))
    lo2, hi2 = g2.min(), g2.max()
    for row in range(60):
        line = ""
        for col in range(60):
            v = (g2[row, col] - lo2) / (hi2 - lo2 + 1e-12)
            k = int(np.clip(v * (len(chars) - 1), 0, len(chars) - 1))
            line += chars[k]
        print("   " + line)
    print(f"   （范围 [{lo2:.5f}, {hi2:.5f}]）")

    # 9 棒候选位置取值扫描：半径 260 与 410 两套
    print()
    print("候选棒位取值（验证棒位半径）:")
    y, x = np.ogrid[:1200, :1200]
    r = np.sqrt((x - 600) ** 2 + (y - 600) ** 2)
    for R in (260, 410):
        # 8 个邻棒位置
        vals = []
        for bi in (-1, 0, 1):
            for bj in (-1, 0, 1):
                if bi == 0 and bj == 0:
                    continue
                c, rr = 600 + bi * R, 600 - bj * R
                m = (x - c) ** 2 + (y - rr) ** 2 <= 60 ** 2
                vals.append(recon_u[m].mean())
        print(f"  R={R} px: 8 邻棒芯均值 {[f'{v:.5f}' for v in vals]}")
    m = (x - 600) ** 2 + (y - 600) ** 2 <= 60 ** 2
    print(f"  中心棒芯: {recon_u[m].mean():.5f}")

    # 空场列轮廓（验证放大率）
    empty = np.load(FLAW / "results" / "empty_proj.npy").astype(np.float64)
    print()
    print("空场 row16 列轮廓（每 60 列均值，验证束斑宽度→放大率）:")
    prof = empty[16].reshape(20, 60).mean(axis=1)
    for i, v in enumerate(prof):
        bar = "#" * int(v / prof.max() * 40)
        print(f"   列 {i*60:4d}-{i*60+59:4d}: {v:7.2f} {bar}")


if __name__ == "__main__":
    main()
