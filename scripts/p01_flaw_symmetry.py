"""P0-1 前置检查第三步（b）：厘清 90° 对称性的真实口径。

第三步（a）暴露矛盾：均匀 3.0MeV 的 max|proj[a]-proj[a+90]| = 0.788，
与既有记录「a000==a090==a180==a270 逐像素差 <=1e-6」不符。本脚本查清：

1. 均匀数据：proj[0] vs proj[90]/[180]/[270] 是否精确相等？哪些角度对不等？
2. 缺陷数据：同样检查；缺陷信号（flaw-unif 同角度差）是否 90° 周期？
3. MC 噪声底：随机角度对的差（无对称性预期）作参照。
4. 逐角度打印 90° 差的最大值分布，定位不等来自哪些角度。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FLAW = ROOT / "3_3_flaw" / "fuel_ct_scan_z0.0_20260804_195102"
UNIF = ROOT / "3_3" / "energy_scan_3.0MeV_20260726_050608_20260726_050609"


def load_flaw(a):
    return np.load(FLAW / f"projections/proj_a{a:06.1f}_z+0.npy").astype(np.float64)


def load_unif(a):
    return np.load(UNIF / f"projections/proj_a{a:03d}_z+0.npy").astype(np.float64)


print("=" * 72)
print("1. 精确相等性检查（均匀 3.0MeV，特定角度对）")
print("=" * 72)
for a in [0, 1, 5, 45, 90]:
    for off in [90, 180, 270]:
        d = np.abs(load_unif(a) - load_unif((a + off) % 360))
        print(f"  unif |proj[{a:3d}]-proj[{(a+off)%360:3d}]|: max={d.max():.6e} "
              f"mean={d.mean():.3e} n_nonzero={(d>0).sum()}")

print()
print("=" * 72)
print("2. 均匀数据 90° 差按角度分布（哪些角度对不等）")
print("=" * 72)
d90_unif = np.zeros(360)
for a in range(360):
    d90_unif[a] = np.abs(load_unif(a) - load_unif((a + 90) % 360)).max()
print(f"  max over a = {d90_unif.max():.6e} at a={int(np.argmax(d90_unif))}")
print(f"  mean over a = {d90_unif.mean():.6e}；中位数 = {np.median(d90_unif):.6e}")
print(f"  ==0 的角度对数: {(d90_unif == 0).sum()}/360；<1e-6: {(d90_unif < 1e-6).sum()}/360")
big = np.where(d90_unif > 1e-3)[0]
print(f"  >1e-3 的角度: {big[:20].tolist()}{'...' if len(big) > 20 else ''} (共 {len(big)})")

print()
print("=" * 72)
print("3. MC 噪声底（无对称预期的随机角度对，均匀数据）")
print("=" * 72)
rng = np.random.default_rng(42)
ds = []
for _ in(0,):
    pass
for a, b in [(3, 200), (17, 141), (88, 301), (250, 12)]:
    d = np.abs(load_unif(a) - load_unif(b))
    ds.append(d.max())
    print(f"  unif |proj[{a:3d}]-proj[{b:3d}]|: max={d.max():.4f} mean={d.mean():.4f}")
print(f"  => 随机角度对 max 参照: {max(ds):.4f}")

print()
print("=" * 72)
print("4. 缺陷数据 90° 差 + 缺陷信号周期性")
print("=" * 72)
d90_flaw = np.zeros(360)
for a in range(360):
    d90_flaw[a] = np.abs(load_flaw(a) - load_flaw((a + 90) % 360)).max()
print(f"  flaw max|proj[a]-proj[a+90]| over a = {d90_flaw.max():.6f} at a={int(np.argmax(d90_flaw))}")
print(f"  mean over a = {d90_flaw.mean():.6f}；中位数 = {np.median(d90_flaw):.6f}")
print(f"  ==0 的角度对数: {(d90_flaw == 0).sum()}/360")
big_f = np.where(d90_flaw > 1.0)[0]
print(f"  >1.0 的角度: {big_f[:20].tolist()}{'...' if len(big_f) > 20 else ''} (共 {len(big_f)})")

# 缺陷信号（flaw-unif 同角度）是否 90° 周期
print()
print("缺陷信号 diff(a) = flaw[a]-unif[a] 的 90° 周期性:")
for a in [0, 7, 23, 45, 100, 200, 333]:
    da = load_flaw(a) - load_unif(a)
    db = load_flaw((a + 90) % 360) - load_unif((a + 90) % 360)
    print(f"  a={a:3d}: max|diff(a)-diff(a+90)| = {np.abs(da-db).max():.4f} "
          f"(diff(a) 自身 max={np.abs(da).max():.2f})")

# 缺陷信号累加（360 角度）定位
print()
print("缺陷信号累加定位（360 角度求和，MC 噪声相消）:")
acc = np.zeros((32, 1200))
for a in range(360):
    acc += load_flaw(a) - load_unif(a)
i = np.unravel_index(np.argmax(np.abs(acc)), acc.shape)
print(f"  max|acc| = {np.abs(acc).max():.1f} at row={i[0]}, col={i[1]} (值 {acc[i]:.1f})")
# 负值区（吸收更强）与正值区（吸收更弱）范围
neg = acc < -np.abs(acc).max() * 0.3
pos = acc > np.abs(acc).max() * 0.3
print(f"  |acc|>30%max 的像素: 负 {neg.sum()} 个 / 正 {pos.sum()} 个")
if neg.sum():
    rs, cs = np.where(neg)
    print(f"  负区行范围 {rs.min()}-{rs.max()}，列范围 {cs.min()}-{cs.max()}")
if pos.sum():
    rs, cs = np.where(pos)
    print(f"  正区行范围 {rs.min()}-{rs.max()}，列范围 {cs.min()}-{cs.max()}")

# 180° 对称性（缺陷数据）
d180 = [np.abs(load_flaw(a) - load_flaw((a + 180) % 360)).max() for a in range(360)]
print(f"  flaw max|proj[a]-proj[a+180]| over a = {max(d180):.6f}")
