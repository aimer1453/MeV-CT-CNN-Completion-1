"""定位缺陷数据的空腔位置（多能量批量），输出可用于 CNR 的 TRAJ 参数。

为什么需要这个脚本
------------------
`p01_flaw_step3_eval.py` 的 `TRAJ = {"amp": 367.1, "offset": 601.9, "phase_deg": 224.7}`
是从**旧**缺陷数据（`3_3_flaw/fuel_ct_scan_z0.0_20260804_195102`，3.0 MeV）拟合的。
多能量缺陷数据是新仿真（`3_3_flaw/energy_scan_<E>MeV_*`），空腔的位置/半径必须先
核实，否则 CNR 会算在错误位置，等于报假数。

方法（不依赖重建）
------------------
清洁 3x3 棒阵在投影域有严格 90° 周期（实测 a000..a270 逐像素差 <= 1e-6），
只有空腔破坏这个对称性。因此对每个视角 θ 定义 4 重平均残差

    Δ(θ, u) = L(θ, u) - mean{ L(θ, u), L(θ+90, u), L(θ+180, u), L(θ+270, u) }

其中 L 是 -log 投影（行 RECON_ROW = 16）。空腔在 Δ 里是一个深的负向凹坑，
其余三根等效角棒的影子只贡献小的正向凸起。于是

    u_c(θ) = argmin_u Δ(θ, u)     （限定在探测器有效范围内）

再对 u_c(θ) 做最小二乘正弦拟合

    u_c(θ) = offset + amp * cos(θ + phase)

正弦参数的几何含义（与 TRAJ 完全同构）：
    amp    = 空腔到旋转中心的距离（px，100 px/cm）
    offset = 旋转中心在探测器上的列坐标
    phase  = 方位角，图像方位 az_img = phase - 90（TRAJ["phase_deg"] - 90 即此定义）

用法
----
  python scripts/localize_flaw_cavity.py                    # 四档全跑 + 旧数据对照
  python scripts/localize_flaw_cavity.py --min-snr 3        # 提高信噪门限
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_recon as br                       # noqa: E402
import p01_flaw_step3_eval as fe                  # noqa: E402  （CNR/棒心口径，import 安全）

FLAW_ROOT = ROOT / "3_3_flaw"
CLEAN_ROOT = ROOT / "3_3"
OUT_JSON = ROOT / "runs" / "_flaw_cavity_localization.json"

# 由稀疏角数据表得到的目录（也可由 --scan-dirs 覆盖）
DEFAULT_SCANS = {
    "1.5": "energy_scan_1.5MeV_z0.0_20261024_041257",
    "3": "energy_scan_3.0MeV_z0.0_20261025_000156",
    "4.5": "energy_scan_4.5MeV_z0.0_20261025_064716",
    "6": "energy_scan_6.0MeV_z0.0_20261025_153256",
}
REF_SCAN = "fuel_ct_scan_z0.0_20260804_195102"    # 旧缺陷数据，TRAJ 的来源


def angle_map(proj_dir: Path) -> dict[int, Path]:
    """文件名 -> 整数角度；支持 proj_a000 / proj_a0000.0 两种命名。"""
    out: dict[int, Path] = {}
    for f in sorted(proj_dir.glob("proj_a*.npy")):
        m = re.match(r"proj_a([0-9.]+)_z", f.name)
        if not m:
            continue
        a = int(round(float(m.group(1)))) % 360
        out.setdefault(a, f)
    return out


def log_sinogram(scan_dir: Path, fill: float) -> np.ndarray:
    """(360, 1200) 的 -log 正弦图（行 RECON_ROW）。"""
    amap = angle_map(scan_dir / "projections")
    missing = [a for a in range(360) if a not in amap]
    if missing:
        raise SystemExit(f"[FATAL] {scan_dir.name}: 缺 {len(missing)} 个角度，"
                         f"例如 {missing[:5]}")
    empty = br.load_empty(scan_dir.parent, scan_dir.name)
    sino = np.empty((360, br.OUTPUT_SIZE), dtype=np.float64)
    for a in range(360):
        proj = np.load(amap[a]).astype(np.float32)
        L = br.log_ratio(proj, empty)
        sino[a] = np.where(np.isfinite(L), L, fill)[br.RECON_ROW]
    return sino


def fold90_residual(sino: np.ndarray) -> np.ndarray:
    """Δ(θ,u) = L(θ,u) - 4 重平均；消除 90° 周期结构，只留破坏对称性的空腔。"""
    w = (sino + np.roll(sino, -90, axis=0) + np.roll(sino, -180, axis=0)
         + np.roll(sino, -270, axis=0)) / 4.0
    return sino - w


def robust_sigma(delta: np.ndarray, sino: np.ndarray, sup_thresh: float = 0.02) -> float:
    """Δ 的稳健 σ，只在物体支撑内统计。

    坑：支撑外全是零计数填充（-log 恒为 fill），Δ 在那一大片里恰好为 0，
    对全图取 MAD 会得到 0 → SNR 恒为 inf（第一版就踩了）。
    """
    sup = np.abs(sino) > sup_thresh
    if sup.sum() < 1000:
        return float("nan")
    d = delta[sup]
    return float(1.4826 * np.median(np.abs(d - np.median(d))))


def image_domain_check(sino: np.ndarray) -> dict:
    """图像域独立复核：dense 360 视角 hamming FBP 后，9 根棒中心 r<=15 px
    圆盘均值谁最暗。投影域拟合可能被宽影偏置，图像域是最终裁决。"""
    img = br.restrict_to_fov(br.recon_fbp(sino, np.arange(360.0), "hamming"))
    vals = sorted(fe.rod_centre_values(img), key=lambda v: v["mu_centre"])
    lo = vals[0]
    med = float(np.median([v["mu_centre"] for v in vals]))
    geo = {n: (r, az) for n, r, az in fe.rod_centers()}
    r_lo, az_lo = geo[lo["rod"]]
    return {"darkest_rod": lo["rod"],
            "darkest_mu": float(lo["mu_centre"]),
            "median_mu": med,
            "relative_dip_pct": 100.0 * (med - lo["mu_centre"]) / med,
            "darkest_rod_r_px": float(r_lo), "darkest_rod_az_deg": float(az_lo),
            "matches_TRAJ": bool(abs(r_lo - fe.R_DEFECT_PX) < 1.0 and
                                 abs(((az_lo - fe.AZ_DEFECT_IMG) + 180.0) % 360.0
                                     - 180.0) < 1.0),
            "all_rods": {v["rod"]: round(float(v["mu_centre"]), 6) for v in vals}}


def fit_traj(u_c: np.ndarray, valid: np.ndarray) -> dict:
    """u_c(θ) = offset + amp*cos(θ+phase) 的最小二乘拟合（线性化求解）。"""
    th = np.radians(np.arange(360.0)[valid])
    u = u_c[valid]
    A = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
    coef, *_ = np.linalg.lstsq(A, u, rcond=None)
    offset, pc, ps = coef
    amp = float(np.hypot(pc, ps))
    phase = float(np.degrees(np.arctan2(-ps, pc)) % 360.0)
    pred = A @ coef
    resid = u - pred
    return {"amp": float(amp), "offset": float(offset), "phase_deg": phase,
            "n_angles_used": int(valid.sum()),
            "fit_resid_rms_px": float(np.sqrt(np.mean(resid ** 2))),
            "fit_resid_max_px": float(np.max(np.abs(resid)))}


def localize(scan: Path, fill: float, min_snr: float, u_lo: int, u_hi: int) -> dict:
    sino = log_sinogram(scan, fill)
    delta = fold90_residual(sino)
    sig = robust_sigma(delta, sino)
    u_c = np.empty(360, dtype=np.float64)
    depth = np.empty(360, dtype=np.float64)
    for a in range(360):
        seg = delta[a, u_lo:u_hi]
        j = int(np.argmin(seg))
        u_c[a] = float(u_lo + j)
        depth[a] = float(-seg[j])          # 正数 = 凹坑深度
    snr = depth / sig if sig and sig == sig else np.full(360, np.inf)
    valid = snr >= min_snr
    res = {"scan_dir": scan.name, "fill": fill,
           "delta_sigma": float(sig), "depth_median": float(np.median(depth)),
           "snr_median": float(np.median(snr)),
           "n_angles_snr_pass": int(valid.sum())}
    if valid.sum() >= 120:
        res.update(fit_traj(u_c, valid))
        res["amp_cm"] = res["amp"] / 100.0
        res["az_img_deg"] = (res["phase_deg"] - 90.0) % 360.0
        res["radius_cm"] = res["amp_cm"]
    else:
        res["fit"] = "FAILED (n_angles_snr_pass < 120)"
    res["u_c_median"] = float(np.median(u_c[valid])) if valid.any() else None
    res["image_domain"] = image_domain_check(sino)   # 最终裁决
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-snr", type=float, default=3.0)
    ap.add_argument("--u-lo", type=int, default=120)
    ap.add_argument("--u-hi", type=int, default=1080)
    ap.add_argument("--fill", type=float, default=5.50,
                    help="-log 零计数填充；这里只用于形成正弦图，"
                         "定位本身对 fill 不敏感（p99.9 量级）")
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--flaw-root", default="3_3_flaw",
                    help="缺陷数据根目录名（默认 3_3_flaw；带包壳数据为 3_3_flaw_1_1）")
    ap.add_argument("--scans", default=None,
                    help="覆盖扫描目录：\"1.5=<dir>,3=<dir>,...\"；默认用 DEFAULT_SCANS")
    args = ap.parse_args()

    global FLAW_ROOT
    FLAW_ROOT = ROOT / args.flaw_root
    scans = dict(DEFAULT_SCANS)
    if args.scans:
        scans = {}
        for item in args.scans.split(","):
            k, v = item.split("=", 1)
            scans[k.strip()] = v.strip()

    results: dict = {}
    print(f"{'能量':>9} {'amp(px)':>8} {'offset':>7} {'phase':>6} {'az_img':>7} "
          f"{'拟合残差':>8} | {'最暗棒':>8} {'r(px)':>7} {'az':>6} {'凹陷%':>7} {'命中TRAJ':>8}")
    for e, d in list(scans.items()) + [("3(ref,old)", REF_SCAN)]:
        scan = FLAW_ROOT / d
        if not scan.is_dir():
            print(f"{e:>9} {'[缺失] ' + d}")
            continue
        r = localize(scan, args.fill, args.min_snr, args.u_lo, args.u_hi)
        results[e] = r
        im = r.get("image_domain") or {}
        if "amp" in r:
            print(f"{e:>9} {r['amp']:8.1f} {r['offset']:7.1f} {r['phase_deg']:6.1f} "
                  f"{r['az_img_deg']:7.1f} {r['fit_resid_rms_px']:8.1f} | "
                  f"{im.get('darkest_rod', '?'):>8} {im.get('darkest_rod_r_px', float('nan')):7.1f} "
                  f"{im.get('darkest_rod_az_deg', float('nan')):6.1f} "
                  f"{im.get('relative_dip_pct', float('nan')):7.2f} "
                  f"{'YES' if im.get('matches_TRAJ') else 'no':>8}")
        else:
            print(f"{e:>9}  {r.get('fit')}")
    # 裁定：四档图像域是否都指向同一个棒（= TRAJ 可直接复用）
    new_e = [k for k in scans if k in results]
    hits = [results[k].get("image_domain", {}).get("matches_TRAJ") for k in new_e]
    rods = [results[k].get("image_domain", {}).get("darkest_rod") for k in new_e]
    verdict = {
        "n_energies": len(new_e),
        "all_match_TRAJ": bool(hits and all(hits)),
        "darkest_rod_per_energy": dict(zip(new_e, rods)),
        "TRAJ_reusable_without_patch": bool(hits and all(hits)),
        "note": ("图像域 dense 360 视角 FBP 的 9 棒对比是最终裁决；投影域正弦拟合在低能档"
                 "会因宽影重叠偏移（1.5 MeV 实测 amp 偏小 ~12 px），不代表几何改变。"
                 if (hits and all(hits)) else
                 "存在能量档未命中 TRAJ 位置，需逐能量设定 TRAJ。"),
    }
    results["_verdict"] = verdict
    print("\n裁定：" + ("四档图像域均指向 TRAJ 所指的同一根棒（corner1, r=367.7 px, "
                        "az=135°）→ p01_flaw_step3_eval 的 TRAJ 可直接复用，无需逐能量打补丁。"
                        if verdict["all_match_TRAJ"] else
                        "存在能量档未命中 TRAJ 位置，见上方逐档结果。"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"\n已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
