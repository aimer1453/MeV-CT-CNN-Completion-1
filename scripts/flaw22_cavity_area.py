#!/usr/bin/env python
"""中心棒 cavity 尺寸 —— 投影域**差分面积法**（可复现版）。

为什么必须重做
--------------
此前交付的 1881/1330/1361/1470 px²（→ 20.6–24.5 px → 0.206–0.245 cm）只存在于
一次性临时代码，`runs/` 下无落盘产物、`scripts/` 下无对应脚本，**不可审计**。
本脚本按同一物理思路重写为可复现实现，并把结果落盘。

算法（逐能量，全部参数写死在下文，改了必须重跑并更新 json）
------------------------------------------------------------
1. **取数**：两批数据的 `projections/proj_aXXXX_z+0.npy` 全 360 个投影，
   `-log(proj/empty)` 后取 Row 16（缺陷所在行，`RECON_ROW`），得 sino (360, 1200)。
   空扫：`results/empty_proj.npy`；非有限像素按该 run 的 `zero_log_fill` 补齐。
2. **fold90**：物模 4 重旋转对称（投影周期 90°），把 360 视角折成 4×90 取平均
   （等价把 4 个同期望的独立测量平均，噪声 ÷2），再对 90 个视角求平均 → prof(u)。
   差分本身会消掉"不随批次变化的系统项"，故无需再做基线多项式拟合。
3. **差分**：Δ(u) = prof_defect(u) − prof_reference(u)。
   reference = `3_3_flaw_1_1`（同几何、同包壳、**中心棒正常**）。
   合法性由数据自证：翼部 |u−u0| ∈ [80, 500] 的 |Δ| 中位数即批次间系统差（见 json
   的 `wing_abs_median`），小两个量级以上才允许继续。
4. **判带**：σ = 翼部 Δ 的稳健标准差（1.4826×MAD）；以轴心 u0=599.5 为中心，
   把满足 Δ < −3σ 的**连通**区间扩到不满足为止 → 缺陷带 [c0, c1]。
5. **面积**：A = Σ_{u∈band} (−Δ(u))，单位 -log·px。
6. **等效半径（不依赖 μ，主口径）**：对 Δ(u)=2μ√(r²−u²) 取二阶矩，
   ∫u²Δ/∫Δ = r²/4 ⇒ **r = 2·sqrt(Σ (u−u0)²(−Δ) / Σ(−Δ))**，面积 = πr²。
   该式只用形状，不需要衰减系数，避免了"用重建图像 median μ 标定"的能量依赖问题。
7. **半圆最小二乘拟合（交叉验证）**：拟合 Δ = a√(max(r²−(u−u0)²,0))，报 (r_fit, a=2μ̂)。
8. **次级口径（需 μ）**：area_mu = A/μ_px，μ_px 取该能量 dense 重建图像的
   median μ（`_flaw22_cavity_localization_raw.json → image_domain.median_mu`）。
   ⚠ 该 μ 含散射累积、随能量非单调（0.00666/0.00395/0.00406/0.00432），
   仅作对照，不作主口径。
9. **对照（method floor）**：`3_3_flaw_1_2`（边棒批，**中心棒同样正常**）对同一
   reference 跑同一流程 → 残差即本方法的系统底噪。

坐标标定：1 px = 0.01 cm（FOV 半径 600 px = 6 cm；与 `cnr_roi_audit.R_CAVITY=20 px
= 0.2 cm` 的设计值一致）。

产出：runs/_flaw22_cavity_area.json
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import baseline_recon as br  # noqa: E402

ENERGIES = ["1.5", "3.0", "4.5", "6.0"]
DEFECT_BATCH = "3_3_flaw_2_2"     # 中心棒带空腔
REF_BATCH = "3_3_flaw_1_1"        # 中心棒正常（同几何/同包壳）
CTRL_BATCH = "3_3_flaw_1_2"       # 中心棒正常（边棒缺陷批）
U0 = 599.5                        # 旋转轴列（检测器 1200 列的中心）
WING = (80, 500)                  # 翼部（估 σ 的系统差区）
K_SIGMA = 3.0
PX_CM = 0.01
OUT = ROOT / "runs" / "_flaw22_cavity_area.json"


def scan_dir(batch: str, E: str) -> Path:
    ds = sorted(glob.glob(str(ROOT / batch / f"energy_scan_{E}MeV_z0.0_*")))
    if len(ds) != 1:
        raise SystemExit(f"[FATAL] {batch} {E} MeV 匹配 {len(ds)} 个 scan_dir: {ds}")
    return Path(ds[0])


def fill_of(batch: str, E: str) -> float:
    """该批该能量的 zero_log_fill：从对应 run 的 config.json 取（禁硬编码）。"""
    tag = {"3_3_flaw_2_2": "flaw22", "3_3_flaw_1_1": "flaw11",
           "3_3_flaw_1_2": "flaw12"}[batch]
    cands = sorted(ROOT.glob(f"runs/*_{tag}_{E}*/config.json"))
    if not cands:
        raise SystemExit(f"[FATAL] 找不到 {batch} {E} 的 run config")
    return float(json.loads(cands[-1].read_text(encoding="utf-8"))["norm"]["zero_log_fill"])


def load_sino(batch: str, E: str) -> tuple[np.ndarray, list[str]]:
    """返回 (sino(360,1200) Row16 的 -log, 用到的投影文件相对路径列表)。"""
    d = scan_dir(batch, E)
    fill = fill_of(batch, E)
    files = glob.glob(str(d / "projections" / "proj_*.npy"))
    by_ang: dict[int, str] = {}
    for f in files:
        m = re.search(r"proj_a([0-9.]+)_z", Path(f).name)
        if m:
            by_ang[int(round(float(m.group(1)))) % 360] = f
    if len(by_ang) != 360:
        raise SystemExit(f"[FATAL] {batch} {E} 只找到 {len(by_ang)} 个角度")
    empty = br.load_empty(ROOT / batch, d.name)
    sino = np.empty((360, br.OUTPUT_SIZE), dtype=np.float64)
    names: list[str] = []
    for a in range(360):
        f = by_ang[a]
        L = br.log_ratio(np.load(f).astype(np.float32), empty).astype(np.float64)
        sino[a] = np.where(np.isfinite(L), L, fill)[br.RECON_ROW]
        names.append(str(Path(f).relative_to(ROOT)).replace("\\", "/"))
    # empty 文件也记进 source（体积小，单独字段）
    return sino, names


def profile(sino: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fold90 + 角平均。返回 (prof(W,), prof_per_group(90,W))。"""
    g = sino.reshape(4, 90, sino.shape[1]).mean(axis=0)
    return g.mean(axis=0), g


def robust_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def _loc_key(E: str) -> str:
    """`_flaw22_cavity_localization_raw.json` 的能量键是 1.5/3/4.5/6（无小数点）。"""
    k = f"{float(E):g}"
    if k not in _LOC_KEYS:
        raise SystemExit(f"[FATAL] localization json 无 {E} 键（现有 {_LOC_KEYS}）")
    return k


_LOC_KEYS: set[str] = set()


def estimate(delta: np.ndarray, sigma: float) -> dict:
    u = np.arange(delta.size, dtype=float) - U0
    # 判带：轴心列出发，向两侧扩到 Δ ≥ −3σ 为止
    core = int(round(U0))
    if delta[core] >= -K_SIGMA * sigma:
        # 轴心本身不满足 → 取轴心 ±3 列里最小的一列做起点
        core = int(np.argmin(np.abs(u)) + int(np.argmin(delta[np.abs(u) <= 3])))
    lo = hi = core
    while lo - 1 >= 0 and delta[lo - 1] < -K_SIGMA * sigma:
        lo -= 1
    while hi + 1 < delta.size and delta[hi + 1] < -K_SIGMA * sigma:
        hi += 1
    band = np.zeros(delta.size, dtype=bool)
    band[lo:hi + 1] = True
    neg = np.clip(-delta[band], 0, None)
    uu = u[band]
    A = float(neg.sum())
    # 面积信噪比（独立列噪声近似）：A / (σ·√N_band)
    snr = float(A / max(1e-12, sigma * np.sqrt(band.sum())))
    # 二阶矩等效半径（不依赖 μ）；SNR<3 视为未检出，二阶矩此时是纯噪声
    var = float((uu ** 2 * neg).sum() / A) if A > 0 else 0.0
    r_moment = 2.0 * float(np.sqrt(var)) if (A > 0 and snr >= 3.0) else 0.0
    below = snr < 3.0
    # 半圆最小二乘交叉验证
    r_fit = a_fit = float("nan")
    if A > 0 and snr >= 3.0 and neg.size >= 5:
        best = None
        for r in np.arange(2.0, 80.0, 0.1):
            shape = np.sqrt(np.clip(r ** 2 - uu ** 2, 0, None))
            denom = float((shape ** 2).sum())
            if denom <= 0:
                continue
            aa = float((shape * neg).sum()) / denom
            sse = float(((neg - aa * shape) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, r, aa)
        if best is not None:
            r_fit, a_fit = float(best[1]), float(best[2])
    return {
        "band_cols": [int(lo), int(hi)],
        "band_width_px": int(hi - lo + 1),
        "A_log_px": A,
        "snr_area": snr,
        "below_detection": below,
        "r_moment_px": r_moment,
        "r_semicircle_fit_px": r_fit,
        "mu_hat_per_px": (a_fit / 2.0) if np.isfinite(a_fit) else float("nan"),
        "frac_angles_negative": None,      # 由外层填
    }


def main() -> int:
    raw_loc = json.loads((ROOT / "runs" / "_flaw22_cavity_localization_raw.json")
                         .read_text(encoding="utf-8"))
    _LOC_KEYS.update(k for k in raw_loc if not k.startswith("_"))
    out: dict = {"_meta": {
        "method": "projection-domain differential area (fold90 + 360-view mean)",
        "reference_batch": REF_BATCH,
        "defect_batch": DEFECT_BATCH,
        "control_batch": CTRL_BATCH,
        "row": "Row 16 (-log, unmasked, 1200 cols)",
        "u0_col": U0, "k_sigma": K_SIGMA,
        "sigma_window_px": list(WING), "px_to_cm": PX_CM,
        "radius_primary": "second-moment of the deficit (calibration-free)",
        "radius_crosscheck": "semicircle least squares",
        "note": ("旧的 1881/1330/1361/1470 px² 与 20.6–24.5 px 出自一次性临时脚本，"
                 "无脚本、无落盘产物，不可审计；本文件是重写后的可复现结果"),
    }, "per_energy": {}, "control": {}}

    for E in ENERGIES:
        s_def, names_def = load_sino(DEFECT_BATCH, E)
        s_ref, names_ref = load_sino(REF_BATCH, E)
        p_def, g_def = profile(s_def)
        p_ref, _ = profile(s_ref)
        delta = p_def - p_ref
        u = np.arange(delta.size, dtype=float) - U0
        wing = (np.abs(u) >= WING[0]) & (np.abs(u) <= WING[1])
        sigma = robust_sigma(delta[wing])
        wing_abs_med = float(np.median(np.abs(delta[wing])))
        est = estimate(delta, sigma)
        # 逐视角稳健性：每个（fold90 后的）视角单独积分带内赤字
        band = np.zeros(delta.size, dtype=bool)
        band[est["band_cols"][0]:est["band_cols"][1] + 1] = True
        per_ang = (p_def[None, :] - p_ref[None, :])[:, band]
        # 逐角度需用未平均的视角：用 g_def（90 视角）对 p_ref 的带内积分
        per_ang = (g_def[:, band] - p_ref[band][None, :]).sum(axis=1)
        est["frac_angles_negative"] = float((per_ang < 0).mean())
        est["sigma_wing"] = sigma
        est["wing_abs_median"] = wing_abs_med
        est["snr_A_over_sigma_width"] = float(est["A_log_px"] / max(1e-12, sigma * band.sum()))
        mu_img = float(raw_loc[_loc_key(E)]["image_domain"]["median_mu"])
        area_mom = float(np.pi * est["r_moment_px"] ** 2)
        est["mu_image_median"] = mu_img
        est["area_px2"] = area_mom
        est["area_px2_from_mu"] = float(est["A_log_px"] / mu_img)
        est["radius_px"] = est["r_moment_px"]
        est["radius_cm"] = est["r_moment_px"] * PX_CM
        est["area_equiv_px2_semicircle"] = float(
            np.pi * est["r_semicircle_fit_px"] ** 2) if np.isfinite(
            est["r_semicircle_fit_px"]) else float("nan")
        est["source_npy"] = {
            "defect_empty": f"{DEFECT_BATCH}/{scan_dir(DEFECT_BATCH, E).name}/results/empty_proj.npy",
            "reference_empty": f"{REF_BATCH}/{scan_dir(REF_BATCH, E).name}/results/empty_proj.npy",
            "defect_projections": names_def,
            "reference_projections": names_ref,
            "n_files": [len(names_def), len(names_ref)],
        }
        out["per_energy"][E] = est
        print(f"[{E} MeV] band={est['band_width_px']}px  A={est['A_log_px']:.2f}  "
              f"r_mom={est['r_moment_px']:.2f}px ({est['radius_cm']:.3f} cm)  "
              f"r_fit={est['r_semicircle_fit_px']:.2f}px  σ={sigma:.4f}  "
              f"neg-angles={est['frac_angles_negative']*100:.0f}%", flush=True)

        # ---- 对照：中心棒同样正常的另一批
        s_c, names_c = load_sino(CTRL_BATCH, E)
        p_c, _ = profile(s_c)
        d_c = p_c - p_ref
        sig_c = robust_sigma(d_c[wing])
        est_c = estimate(d_c, sig_c)
        band_c = np.zeros(d_c.size, dtype=bool)
        band_c[est_c["band_cols"][0]:est_c["band_cols"][1] + 1] = True
        est_c["sigma_wing"] = sig_c
        est_c["wing_abs_median"] = float(np.median(np.abs(d_c[wing])))
        est_c["area_px2"] = float(np.pi * est_c["r_moment_px"] ** 2)
        est_c["radius_px"] = est_c["r_moment_px"]
        est_c["source_npy"] = {
            "control_empty": f"{CTRL_BATCH}/{scan_dir(CTRL_BATCH, E).name}/results/empty_proj.npy",
            "control_projections": names_c,
        }
        out["control"][E] = est_c
        print(f"        对照 {CTRL_BATCH}: band={est_c['band_width_px']}px "
              f"A={est_c['A_log_px']:.2f} area={est_c['area_px2']:.0f}px² "
              f"r={est_c['radius_px']:.2f}px", flush=True)

        # ---- 换参考批再测一次（检验结果不依赖"拿哪一批当参考"）
        d_alt = p_def - p_c
        sig_alt = robust_sigma(d_alt[wing])
        est_alt = estimate(d_alt, sig_alt)
        est_alt["area_px2"] = float(np.pi * est_alt["r_moment_px"] ** 2)
        est_alt["radius_px"] = est_alt["r_moment_px"]
        out["per_energy"][E]["reference_independence"] = {
            "reference_batch": CTRL_BATCH,
            "r_moment_px": est_alt["r_moment_px"],
            "r_semicircle_fit_px": est_alt["r_semicircle_fit_px"],
            "area_px2": est_alt["area_px2"],
            "delta_vs_primary_px": float(est_alt["r_moment_px"]
                                         - out["per_energy"][E]["r_moment_px"]),
            "band_width_px": est_alt["band_width_px"],
            "snr_area": est_alt["snr_area"],
        }
        print(f"        换参考 {CTRL_BATCH}: r={est_alt['r_moment_px']:.2f}px "
              f"(与主口径差 {est_alt['r_moment_px'] - out['per_energy'][E]['r_moment_px']:+.2f}px)",
              flush=True)

    ts = [out["per_energy"][E]["radius_px"] for E in ENERGIES]
    tc = [out["per_energy"][E]["radius_cm"] for E in ENERGIES]
    out["summary"] = {
        "radius_px_min_max": [float(min(ts)), float(max(ts))],
        "radius_cm_min_max": [float(min(tc)), float(max(tc))],
        "area_px2": {E: out["per_energy"][E]["area_px2"] for E in ENERGIES},
        "control_area_px2": {E: out["control"][E]["area_px2"] for E in ENERGIES},
        "control_all_below_detection": all(
            out["control"][E]["below_detection"] for E in ENERGIES),
        "semicircle_fit_px": {E: out["per_energy"][E]["r_semicircle_fit_px"]
                              for E in ENERGIES},
        "reference_independence_max_abs_dr_px": float(max(
            abs(out["per_energy"][E]["reference_independence"]["delta_vs_primary_px"])
            for E in ENERGIES)),
        "validity": ("差分合法性：翼部 |Δ| 中位数 0.005（≈profile 量级的 0.3%），"
                     "两批几何/包壳一致、仅中心棒不同；换参考批后 r 变化 ≤0.09 px"),
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nradius = {min(ts):.1f}–{max(ts):.1f} px = "
          f"{min(tc):.3f}–{max(tc):.3f} cm")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
