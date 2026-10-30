"""P0-1 补充诊断：flaw/unif 计数比的列分布（无采样假设，无 recon 介入）。

矛盾待解：
- recon 域：正常角棒芯差分 ≈ +0.0001（正常），缺陷角棒 -0.00335，中心棒 -0.00761
- 投影域 3px 采样：缺陷棒 +56.5%，对角"正常"棒 +32.3%（应 ≈0）
- 总计数 +14.2%（360 角全像素）

本脚本直接测：
1. 每列计数比（360 角平均）→ 哪些列增强，9 棒列是否都增强
2. 单角度 a000 row16 比值曲线 → 缺陷/正常棒的投影域足迹
3. 总计数比按区域分解（棒芯/包壳/间隙）→ +14.2% 从哪来
4. flaw 空扫 vs 均匀空扫已证逐位相同 → 差异全在物体
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"

angles = list(range(360))


def main():
    print("加载 flaw 360 投影...", flush=True)
    pf = np.stack([np.load(FLAW / "projections" / f"proj_a{a:06.1f}_z+0.npy").astype(np.float64)
                   for a in angles])
    print("加载 unif 360 投影...", flush=True)
    pu = np.stack([np.load(UNIF / "projections" / f"proj_a{a:03d}_z+0.npy").astype(np.float64)
                   for a in angles])
    print(f"flaw shape {pf.shape}, unif shape {pu.shape}")
    print(f"总计数: flaw {pf.sum():.3e}  unif {pu.sum():.3e}  比 {pf.sum()/pu.sum():.4f}")

    # ---- 1. 每列计数比（360 角平均，row16）----
    ratio_col = (pf[:, 16, :].sum(0) + 1e-9) / (pu[:, 16, :].sum(0) + 1e-9)
    print("\n[1] row16 每列计数比（360 角累加）关键列:")
    for c in [50, 200, 340, 345, 500, 600, 700, 855, 860, 1000, 1150]:
        print(f"    col {c:4d}: {ratio_col[c]:.4f}")
    # 按 50px bin 打印
    print("    50px bin 比值:")
    for c0 in range(0, 1200, 100):
        print(f"      [{c0:4d},{c0+100:4d}): {ratio_col[c0:c0+100].mean():.4f}")

    # ---- 2. 单角度 a000 row16 比值曲线（9 棒位置）----
    r0 = (pf[0, 16, :] + 1e-9) / (pu[0, 16, :] + 1e-9)
    print("\n[2] a000 row16 单角度比值（9 棒谷位列附近）:")
    for c in [92, 185, 337, 340, 469, 596, 600, 723, 746, 857, 860, 1095]:
        print(f"    col {c:4d}: {r0[c]:.4f}")
    # 9 棒芯 5px 均值（unif 谷位: ±263, ±131, ±4, +123, +146, +257, +495 的对称子集）
    cores = [340, 600, 860, 470, 730, 260, 940]  # ±260, 0, ±130(边棒内?), ...
    print("    候选棒芯 5px 均值:",
          {c: round(float(r0[c-2:c+3].mean()), 4) for c in [340, 600, 860]})

    # ---- 3. 区域分解（全 360 角，全 32 行）----
    print("\n[3] 总计数比按区域分解（全 360 角全 32 行）")
    # 棒芯列集合（a000 视图）: ±260, 0 → 但旋转下每列都曾是棒芯。用 recon 域掩膜更稳：
    # 这里简单按 |col-600| 分箱
    cols = np.arange(1200)
    for name, sel in [("|col-600|<60 (中心区)", np.abs(cols - 600) < 60),
                      ("240<|col-600|<280 (外侧棒道)", (np.abs(cols - 600) > 240) & (np.abs(cols - 600) < 280)),
                      ("100<|col-600|<160 (内侧棒道)", (np.abs(cols - 600) > 100) & (np.abs(cols - 600) < 160)),
                      ("其他", ~((np.abs(cols - 600) < 60) | ((np.abs(cols - 600) > 240) & (np.abs(cols - 600) < 280)) | ((np.abs(cols - 600) > 100) & (np.abs(cols - 600) < 160))))]:
        sf, su = pf[:, :, sel].sum(), pu[:, :, sel].sum()
        print(f"    {name}: flaw {sf:.4e} unif {su:.4e} 比 {sf/su:.4f}  像素占比 {sel.mean()*100:.1f}%")

    # ---- 4. 行方向（z）检查 ----
    print("\n[4] 行方向计数比（每行 360 角×1200 列均值）:")
    rr = pf.sum((0, 2)) / pu.sum((0, 2))
    for z in [0, 8, 16, 24, 31]:
        print(f"    row {z:2d}: {rr[z]:.4f}")


if __name__ == "__main__":
    main()
