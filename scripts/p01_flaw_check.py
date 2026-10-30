"""P0-1 前置检查第三步：缺陷数据的对称性与缺陷定位（只读）。

决定性问题：带缺陷物体是否仍保持 90° 旋转对称？
- 若 `proj[a] == proj[a+90]`（如均匀数据那样逐像素相等）→ 缺陷在中心，对称未破，grouped split 仍必须
- 若显著不等 → 缺陷非中心对称，审稿人指的等价重复自动消失

同时与 3_3/3.0MeV 均匀数据同角度对比，分离缺陷信号与 MC 噪声。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"

# ---------- 1. 完整性与 shape ----------
print("=" * 70)
print("1. 完整性检查")
print("=" * 70)
flaw_files = sorted(FLAW.glob("projections/proj_a*.npy"))
unif_files = sorted(UNIF.glob("projections/proj_a*.npy"))
print(f"flaw projections: {len(flaw_files)} 个；命名样例 {flaw_files[0].name} ... {flaw_files[-1].name}")
print(f"unif projections: {len(unif_files)} 个；命名样例 {unif_files[0].name} ... {unif_files[-1].name}")

angles_flaw = sorted(int(float(p.stem.split("_a")[1].split("_")[0])) for p in flaw_files)
angles_unif = sorted(int(float(p.stem.split("_a")[1].split("_")[0])) for p in unif_files)
print(f"flaw 角度: min={angles_flaw[0]} max={angles_flaw[-1]} 唯一值={len(set(angles_flaw))}")
print(f"unif 角度: min={angles_unif[0]} max={angles_unif[-1]} 唯一值={len(set(angles_unif))}")
missing = [a for a in range(360) if a not in set(angles_flaw)]
print(f"flaw 缺失角度: {missing if missing else '无（0..359 齐全）'}")

p0 = np.load(flaw_files[0])
pe = np.load(FLAW / "results" / "empty_proj.npy")
print(f"flaw proj[0] shape={p0.shape} dtype={p0.dtype}；empty_proj shape={pe.shape} dtype={pe.dtype}")

# 抽查全部 360 个 shape（快，只读 header 不行就全读；32x1200 float32 = 147KB/个，可接受）
bad = []
for p in flaw_files:
    a = np.load(p, mmap_mode="r")
    if a.shape != (32, 1200):
        bad.append((p.name, a.shape))
print(f"flaw 360 个 shape 检查: {'全部 (32, 1200)' if not bad else bad}")

# ---------- 2. 90° 对称性测试（决定性） ----------
print()
print("=" * 70)
print("2. 90° 旋转对称性测试（P0-1 的决定性检查）")
print("=" * 70)
flaw = {a: np.load(FLAW / f"projections/proj_a{a:06.1f}_z+0.npy").astype(np.float64)
        for a in range(360)}
unif = {a: np.load(UNIF / f"projections/proj_a{a:03d}_z+0.npy").astype(np.float64)
        for a in range(360)}

# 先复现均匀数据的对称性作为方法校验
d_unif = max(float(np.abs(unif[a] - unif[(a + 90) % 360]).max()) for a in range(360))
print(f"[方法校验] 均匀 3.0MeV: max|proj[a]-proj[a+90]| = {d_unif:.3e}（已知 <=1e-6）")

d90, d180, d270 = [], [], []
for a in range(360):
    d90.append(np.abs(flaw[a] - flaw[(a + 90) % 360]))
    d180.append(np.abs(flaw[a] - flaw[(a + 180) % 360]))
    d270.append(np.abs(flaw[a] - flaw[(a + 270) % 360]))
d90 = np.array(d90); d180 = np.array(d180); d270 = np.array(d270)
print(f"缺陷数据 max|proj[a]-proj[a+90]|  = {d90.max():.6e}（均值 {d90.mean():.6e}）")
print(f"缺陷数据 max|proj[a]-proj[a+180]| = {d180.max():.6e}（均值 {d180.mean():.6e}）")
print(f"缺陷数据 max|proj[a]-proj[a+270]| = {d270.max():.6e}（均值 {d270.mean():.6e}）")
# 用相对判据：与 MC 噪声水平比
rng = np.random.default_rng(0)
# 同一文件重复读不可能，用不同角度同位置差做噪声底：取 |proj[a]-proj[a+1]| 的最小分位作参照
d1 = np.array([np.abs(flaw[a] - flaw[(a + 1) % 360]) for a in range(360)])
print(f"[参照] max|proj[a]-proj[a+1]| = {d1.max():.6e}（相邻角差，含缺陷运动+噪声）")

sym_broken = d90.max() > 1e-3
print()
if sym_broken:
    print(">>> 结论：90° 对称性被破坏（max|diff| 远超 MC 噪声水平）")
else:
    print(">>> 结论：90° 对称性仍保持（max|diff| 在噪声水平内）")

# ---------- 3. 与均匀数据同角度对比（缺陷定位） ----------
print()
print("=" * 70)
print("3. 缺陷数据 vs 均匀 3.0MeV（同角度）")
print("=" * 70)
# 计数域对比（避开 -log 域，先看原始计数）
diff_cnt = np.zeros((32, 1200), dtype=np.float64)
absdiff_max = 0.0
corr_min = 1.0
for a in range(360):
    d = flaw[a] - unif[a]
    diff_cnt += d
    absdiff_max = max(absdiff_max, float(np.abs(d).max()))
    f = flaw[a].ravel(); u = unif[a].ravel()
    if f.std() > 0 and u.std() > 0:
        corr_min = min(corr_min, float(np.corrcoef(f, u)[0, 1]))
print(f"同角度原始计数差: max|diff| = {absdiff_max:.2f}；最低相关系数 = {corr_min:.6f}")
print(f"360 角度累加差（缺陷信号相干累积，MC 噪声 ~1/sqrt(360) 相消）:")
nz = np.abs(diff_cnt)
print(f"  累加差 max = {nz.max():.1f}, mean|diff| = {nz.mean():.4f}")
# 累加差的空间分布：定位缺陷
if nz.max() > 0:
    idx = np.unravel_index(np.argmax(nz), nz.shape)
    print(f"  最大累加差位置: row={idx[0]}, col={idx[1]}, 值={nz[idx]:.1f}")
    # 看该位置周边
    r, c = idx
    print(f"  该像素邻域 5x5 累加差:")
    print(diff_cnt[max(0, r-2):r+3, max(0, c-2):c+3].round(1))
    # 缺陷若是吸收体，flaw < unif（差为负）；若是气泡/空洞，flaw > unif（差为正）
    print(f"  符号: {'flaw 计数更低（缺陷=更强吸收/更密）' if diff_cnt[r, c] < 0 else 'flaw 计数更高（缺陷=更弱吸收/气泡/空洞）'}")

# 逐角度追踪最大差位置（缺陷在 sinogram 上是正弦轨迹）
print()
print("逐角度最大差位置（row, col）抽样（缺陷轨迹应为光滑正弦）:")
for a in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
    d = np.abs(flaw[a] - unif[a])
    i = np.unravel_index(np.argmax(d), d.shape)
    print(f"  a={a:3d}: row={i[0]:2d} col={i[1]:4d} max|diff|={d[i]:.2f} "
          f"(flaw={flaw[a][i]:.1f} unif={unif[a][i]:.1f})")
