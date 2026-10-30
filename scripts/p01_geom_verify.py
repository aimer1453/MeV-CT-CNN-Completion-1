"""P0-1 第一步补充：flaw 数据角度序列验证 + 宏全量 diff + 放大率实测。

三个问题：
1. flaw 360 个投影文件名的角度是否均匀 1° 步进（p01_flaw_localize.py 假设 range(360)）
2. flaw 宏 vs 均匀 3.0MeV 宏是否逐字一致（source 几何）
3. 正弦图棒迹位置/间距/阴影边界 → 实测放大率（几何代码 M=1.576 vs 实测 ~1.0 矛盾）

只读，不写任何数据目录。
"""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"


def flaw_angles():
    files = sorted((FLAW / "projections").glob("proj_a*_z+0.npy"))
    angs = np.array([float(re.match(r"proj_a([0-9.]+)_z", f.name).group(1)) for f in files])
    print(f"[1] flaw 投影文件数: {len(files)}")
    print(f"    角度范围 [{angs.min()}, {angs.max()}]，跨度 {angs.max()-angs.min()}")
    steps = np.diff(angs)
    print(f"    步进: min {steps.min():.4f}  max {steps.max():.4f}  "
          f"mean {steps.mean():.4f}  std {steps.std():.4f}")
    print(f"    与整数度最大偏差 {np.abs(angs - np.round(angs)).max():.4f}")
    print(f"    前 12 个角度: {angs[:12]}")
    return angs


def macro_diff():
    print("\n[2] 宏文件全量对比（flaw vs 均匀 3.0MeV）")
    fmac = sorted((FLAW / "macros").glob("scan_a*.mac"))
    umac_dir = UNIF / "macros"

    def norm(text):
        out = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            ln = ln.replace("0.0 cm", "0 cm").replace(" 0.0 ", " 0 ")
            out.append(ln)
        return out

    same = diff = missing = 0
    diff_examples = []
    for f in fmac:
        a = float(re.match(r"scan_a([0-9.]+)_z", f.name).group(1))
        u = umac_dir / f"scan_a{int(round(a)):03d}_z+0.mac"
        if not u.exists():
            missing += 1
            continue
        nf, nu = norm(f.read_text()), norm(u.read_text())
        if nf == nu:
            same += 1
        else:
            diff += 1
            if len(diff_examples) < 3:
                diff_examples.append((f.name, nf, nu))
    print(f"    相同(归一化后): {same}  不同: {diff}  均匀侧缺失: {missing}  共 {len(fmac)}")
    for name, nf, nu in diff_examples:
        print(f"    DIFF {name}:")
        for x, y in zip(nf, nu):
            if x != y:
                print(f"      flaw: {x!r}  unif: {y!r}")
    # 几何相关行汇总（取 flaw 第一个宏）
    first = norm(fmac[0].read_text())
    print("    flaw 宏几何相关行:")
    for ln in first:
        if ln.startswith("/gps"):
            print(f"      {ln}")


def magnification():
    print("\n[3] 放大率实测（row16 正弦图）")
    empty = np.load(UNIF / "results" / "empty_proj.npy").astype(np.float64)
    e16 = empty[16]
    print(f"    empty row16: mean {e16.mean():.1f}  min {e16.min():.1f}  max {e16.max():.1f}  "
          f"| 中心±50px mean {e16[550:650].mean():.1f}  边缘±50px mean "
          f"{np.concatenate([e16[:50], e16[-50:]]).mean():.1f}")

    for tag, d, fmt in [("flaw a000", FLAW, "float"), ("unif a000", UNIF, "int"),
                        ("unif a090", UNIF, "int90")]:
        if fmt == "float":
            proj = np.load(d / "projections" / "proj_a0000.0_z+0.npy").astype(np.float64)
        elif fmt == "int":
            proj = np.load(d / "projections" / "proj_a000_z+0.npy").astype(np.float64)
        else:
            proj = np.load(d / "projections" / "proj_a090_z+0.npy").astype(np.float64)
        row = proj[16]
        # 谷位：局部极小且低于邻域均值
        valleys = []
        for c in range(3, 1197):
            if row[c] < row[c-1] and row[c] <= row[c+1] and row[c] < row[c-40:c+40].mean():
                valleys.append(c)
        # 合并相近谷位
        merged = []
        for c in valleys:
            if merged and c - merged[-1][-1] <= 12:
                merged[-1].append(c)
            else:
                merged.append([c])
        vpos = [int(np.mean(g)) for g in merged]
        # 阴影边界：obj < 0.98*empty 的连续区域
        shadow = row < 0.98 * e16
        idx = np.where(shadow)[0]
        if len(idx):
            # 取包含中心的最大连续段
            segs = []
            start = idx[0]
            for k in range(1, len(idx)):
                if idx[k] != idx[k-1] + 1:
                    segs.append((start, idx[k-1]))
                    start = idx[k]
            segs.append((start, idx[-1]))
            segs = [s for s in segs if s[0] <= 600 <= s[1]] or segs
            seg = max(segs, key=lambda s: s[1] - s[0])
        else:
            seg = (None, None)
        print(f"    {tag}: 谷位列 {vpos}（距中心 {[v-600 for v in vpos]}）")
        print(f"      阴影区 [{seg[0]}, {seg[1]}] → 半宽 "
              f"{(seg[1]-seg[0])/2:.0f} px = {(seg[1]-seg[0])/2*0.01:.2f} cm"
              f"（M=1 预期 360px/3.6cm；M=1.576 预期 568px/5.68cm）")


def main():
    flaw_angles()
    macro_diff()
    magnification()


if __name__ == "__main__":
    sys.exit(main())
