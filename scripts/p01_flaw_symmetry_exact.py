"""P0-1 前置检查第三步（l）：90°/180° 自差异的精确逐像素均值（两扫描对照）。"""
import numpy as np
from pathlib import Path

UNIF = Path("3_3/energy_scan_3.0MeV_20260726_050608_20260726_050609")
FLAW = Path("3_3_flaw/fuel_ct_scan_z0.0_20260804_195102")

pu = {a: np.load(UNIF / f"projections/proj_a{a:03d}_z+0.npy").astype(np.float64)
      for a in range(360)}
pf = {a: np.load(FLAW / f"projections/proj_a{a:06.1f}_z+0.npy").astype(np.float64)
      for a in range(360)}

for tag, p in [("unif(3_3/3.0MeV)", pu), ("flaw(3_3_flaw/)", pf)]:
    s90 = s180 = n = 0.0
    for a in range(360):
        s90 += float(np.abs(p[a] - p[(a + 90) % 360]).sum())
        s180 += float(np.abs(p[a] - p[(a + 180) % 360]).sum())
        n += p[a].size
    print(f"{tag}: mean|proj[a]-proj[a+90]|  = {s90/n:.6f} counts/px")
    print(f"{tag}: mean|proj[a]-proj[a+180]| = {s180/n:.6f} counts/px")
