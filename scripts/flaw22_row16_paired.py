#!/usr/bin/env python
"""中心棒批（3_3_flaw_2_2）Row-16 逐视角配对主指标 —— 与 §5.6 同口径。

口径（严格对齐 §5.6 / dose_bootstrap.py）
--------------------------------------
* 只取 **test 视角**：4 能量 × 45 = **180 个 (energy, view) 配对观测**；
* 每个视角的 Row-16 RMSE = sqrt(mean(误差² over 1200 列))，**unmasked**
  （n_valid = 1200，与 §5.6 的 per_sample_test 一致）；
* CNN 每视角取 **3 个种子 per-view RMSE 的算术平均**（随机性只来视角重采样）；
* linear = 0.5*(sino[a-1] + sino[a+1]) − sino[a]，与 `recon_all()` 逐字同式；
* 配对 bootstrap 10000 次、seed 20261022（与 §5.6 完全同参数）。

符号约定
--------
Δ = CNN − linear（**正 = CNN 更差**，与 §5.6 一致）
ratio = mean(CNN)/mean(linear) = 1 + Δ/mean(linear)
improvement% = −Δ/mean(linear) × 100（**正 = CNN 更好**）

自检（fail-fast）
-----------------
① per-energy 全视角（360 视角）均值必须复现 R59 表
   0.122467/0.069223/0.058827/0.056976（CNN）与
   0.125762/0.059673/0.053077/0.053607（linear）；
② 复现错序检查：corr(per_view, |sin θ|) 必须与 image_metrics.json 的
   corr_rmse_vs_defect_speed 一致（误差图行与角度的映射若错位必失败）。

产出：runs/_flaw22_row16_paired.json
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import baseline_recon as br  # noqa: E402

RUNS = ROOT / "runs"
CSV_DIR = ROOT / "csv_data"
ENERGIES = ["1.5", "3.0", "4.5", "6.0"]
N_BOOT = 10000
SEED = 20261022
EPS_TAB = 5e-6      # R59 表复现容差（zero_log_fill 逐能量不同，必须从 config 读）

# R59 已发布的 per-energy 值（**全 360 视角**聚合），用于自检
REF_R59 = {
    "1.5": (0.122467, 0.125762),
    "3.0": (0.069223, 0.059673),
    "4.5": (0.058827, 0.053077),
    "6.0": (0.056976, 0.053607),
}


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def build_sino(rows: list[dict], fill: float) -> tuple[np.ndarray, np.ndarray]:
    """返回 (sino(360,1200) row16 的 -log, theta(360,))。"""
    scan_dirs = {r["scan_dir"] for r in rows}
    if len(scan_dirs) != 1:
        raise SystemExit(f"[FATAL] 多个 scan_dir: {scan_dirs}")
    root = (ROOT / rows[0]["target_file"]).parent.parent.parent
    empty = br.load_empty(root, scan_dirs.pop())
    sino = np.empty((len(rows), br.OUTPUT_SIZE), dtype=np.float64)
    theta = np.empty(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        proj = np.load(ROOT / r["target_file"]).astype(np.float32)
        if proj.shape != (32, br.OUTPUT_SIZE):
            raise SystemExit(f"[FATAL] 投影形状 {proj.shape} != (32,1200)")
        L = br.log_ratio(proj, empty).astype(np.float64)
        sino[i] = np.where(np.isfinite(L), L, fill)[br.RECON_ROW]
        theta[i] = float(r["target_angle_deg"])
    return sino, theta


def seed_dirs(E: str) -> list[Path]:
    ds = sorted(d for d in RUNS.glob(f"*_flaw22_{E}_flaw_step3") if d.is_dir())
    if len(ds) != 3:
        raise SystemExit(f"[FATAL] {E} MeV 期望 3 个 seed 目录，实得 {len(ds)}: {ds}")
    return ds


def err_row_index(a: int) -> tuple[str, int]:
    """角度 → (误差图相位, 行号)。errs[in_phase] 存的是**缺失相位**（预测视角）的误差：
    phase A = 偶角 (0,2,...,358) → errs['B']；phase B = 奇角 → errs['A']。"""
    if a % 2 == 1:
        return "A", (a - 1) // 2
    return "B", a // 2


def main() -> int:
    per_energy: dict[str, dict] = {}
    pooled_cnn, pooled_lin = [], []

    for E in ENERGIES:
        csv_p = CSV_DIR / f"sparse_targets_flaw22_{E}.csv"
        rows = list(csv.DictReader(open(csv_p, newline="", encoding="utf-8")))
        if len(rows) != 360:
            raise SystemExit(f"[FATAL] {csv_p.name} 行数 {len(rows)} != 360")
        # 3 个 seed 的 fill 必须一致（同批同能量）；fill 逐能量不同，禁硬编码
        ds = seed_dirs(E)
        fills = {float(json.loads((d / "config.json").read_text(encoding="utf-8"))
                       ["norm"]["zero_log_fill"]) for d in ds}
        if len(fills) != 1:
            raise SystemExit(f"[FATAL] {E} MeV 三种子 fill 不一致: {fills}")
        fill = fills.pop()
        sino, theta = build_sino(rows, fill)
        idx = {int(round(float(t))): i for i, t in enumerate(theta)}
        if len(idx) != 360:
            raise SystemExit(f"[FATAL] theta 非 360 唯一角: {len(idx)}")

        # ---- linear：逐视角 row16 误差（全 360 视角一次算完，test 子集后面取）
        lin_pv = np.empty(360, dtype=np.float64)
        for i, a in enumerate(np.round(theta).astype(np.int64)):
            lo, hi = sino[idx[(int(a) - 1) % 360]], sino[idx[(int(a) + 1) % 360]]
            lin_pv[int(a)] = rms(0.5 * (lo + hi) - sino[idx[int(a)]])

        # ---- CNN：3 个 seed 的 per-view RMSE 均值
        cnn_stack = []
        corr_checks = []
        for d in ds:
            eA = np.load(d / "proj_error_map_A.npy")
            eB = np.load(d / "proj_error_map_B.npy")
            if eA.shape != (180, br.OUTPUT_SIZE) or eB.shape != (180, br.OUTPUT_SIZE):
                raise SystemExit(f"[FATAL] {d.name} 误差图形状异常 {eA.shape} {eB.shape}")
            pv_a = np.sqrt(np.square(eA.astype(np.float64)).mean(axis=1))
            pv_b = np.sqrt(np.square(eB.astype(np.float64)).mean(axis=1))
            arr = np.empty(360, dtype=np.float64)
            for a in range(360):
                ph, i = err_row_index(a)
                arr[a] = (pv_a if ph == "A" else pv_b)[i]
            cnn_stack.append(arr)
            # 错序检查：corr(per_view, |sin θ|) 应与 json 一致（中心棒 TRAJ amp=0，phase=0）
            jm = json.loads((d / "image_metrics.json").read_text(encoding="utf-8"))
            for ph, angs, pv in (("A", np.arange(1, 360, 2), pv_a),
                                 ("B", np.arange(0, 360, 2), pv_b)):
                r_json = jm["projection_domain"]["per_view_row16"][ph][
                    "corr_rmse_vs_defect_speed"]["pearson_r"]
                sp = np.abs(np.sin(np.radians(angs + 0.0)))
                r_me = float(np.corrcoef(pv, sp)[0, 1])
                if abs(r_me - r_json) > 2e-3:
                    raise SystemExit(
                        f"[FATAL] {d.name} 相位{ph} 角度映射错序："
                        f"corr={r_me:.6f} vs json {r_json:.6f}")
                corr_checks.append({"run": d.name, "phase": ph,
                                    "corr_recomputed": r_me, "corr_json": r_json})
        cnn_pv = np.mean(np.stack(cnn_stack, axis=0), axis=0)

        # ---- 自检：全视角均值必须复现 R59 表
        ref_cnn, ref_lin = REF_R59[E]
        cnn_all, lin_all = float(cnn_pv.mean()), float(lin_pv.mean())
        if abs(cnn_all - ref_cnn) > EPS_TAB or abs(lin_all - ref_lin) > EPS_TAB:
            raise SystemExit(
                f"[FATAL] {E} MeV 全视角均值未复现 R59："
                f"CNN {cnn_all:.6f} vs {ref_cnn:.6f}；lin {lin_all:.6f} vs {ref_lin:.6f}")

        test_ang = sorted({int(float(r["target_angle_deg"]))
                           for r in rows if r["split"] == "test"})
        if len(test_ang) != 45:
            raise SystemExit(f"[FATAL] {E} MeV test 视角 {len(test_ang)} != 45")
        ti = np.array(test_ang, dtype=int)
        c_t, l_t = cnn_pv[ti], lin_pv[ti]
        pooled_cnn.append(c_t)
        pooled_lin.append(l_t)
        # 逐能量配对统计（n=45，仅供逐能量叙述，主指标仍是 180 视角池化）
        dd = c_t - l_t
        d_m = float(dd.mean())
        d_s = float(dd.std(ddof=1))
        t_e = d_m / (d_s / math.sqrt(len(dd))) if d_s > 0 else float("nan")
        try:
            from scipy import stats as _st
            p_e = float(2.0 * _st.t.sf(abs(t_e), len(dd) - 1))
        except Exception:
            p_e = float(math.erfc(abs(t_e) / math.sqrt(2.0)))
        rb = np.random.default_rng(SEED)
        bi = rb.integers(0, len(dd), size=(N_BOOT, len(dd)))
        r_ci = np.percentile(c_t[bi].mean(axis=1) / l_t[bi].mean(axis=1), [2.5, 97.5])
        per_energy[E] = {
            "n_test_views": 45,
            "cnn_row16_test_mean": float(c_t.mean()),
            "cnn_row16_test_sd": float(c_t.std(ddof=1)),
            "linear_row16_test_mean": float(l_t.mean()),
            "ratio_test": float(c_t.mean() / l_t.mean()),
            "ratio_test_ci95": [float(r_ci[0]), float(r_ci[1])],
            "delta_test": d_m,
            "paired_t": float(t_e), "p_two_sided": p_e,
            "cnn_row16_all_views_mean": cnn_all,
            "linear_row16_all_views_mean": lin_all,
            "ratio_all_views": float(cnn_all / lin_all),
            "r59_reference": {"cnn": ref_cnn, "linear": ref_lin},
            "test_angles": test_ang,
        }
        print(f"[{E} MeV] test: CNN={c_t.mean():.6f} lin={l_t.mean():.6f} "
              f"ratio={c_t.mean()/l_t.mean():.4f} p={p_e:.3g} | all-view ratio="
              f"{cnn_all/lin_all:.4f}（R59 复现 ✓）", flush=True)

    # ================= 三个聚合口径（全部在 test 视角上） =================
    cnn = np.concatenate(pooled_cnn)
    lin = np.concatenate(pooled_lin)
    n = cnn.size
    if n != 180:
        raise SystemExit(f"[FATAL] pooled n={n} != 180")

    delta = cnn - lin
    mean_cnn, mean_lin = float(cnn.mean()), float(lin.mean())
    d_mean = float(delta.mean())
    ratio = mean_cnn / mean_lin
    imp = -d_mean / mean_lin * 100.0
    sd = float(delta.std(ddof=1))
    dz = d_mean / sd

    # ---- 配对 bootstrap（与 dose_bootstrap.py 同参数）
    rng = np.random.default_rng(SEED)
    bidx = rng.integers(0, n, size=(N_BOOT, n))
    cb, lb = cnn[bidx], lin[bidx]
    db = cb - lb
    mb = db.mean(axis=1)
    ratio_b = cb.mean(axis=1) / lb.mean(axis=1)
    imp_b = -mb / lb.mean(axis=1) * 100.0
    sdb = db.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dz_b = np.where(sdb > 0, mb / sdb, np.nan)

    def ci(v, lo=2.5, hi=97.5):
        return float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))

    prim = {
        "definition": "equal-weight paired over 180 (energy, view) test observations",
        "n": n,
        "mean_cnn": mean_cnn, "mean_linear": mean_lin,
        "delta_cnn_minus_linear": d_mean,
        "delta_ci95": list(ci(mb)),
        "ratio_cnn_over_linear": ratio,
        "ratio_ci95": list(ci(ratio_b)),
        "improvement_pct": imp,
        "improvement_ci95": list(ci(imp_b)),
        "cohens_dz": dz,
        "dz_ci95": list(ci(dz_b)),
        "paired_t": None,
        "n_boot": N_BOOT, "seed": SEED,
    }
    # 配对 t 检验（双侧，df = n−1），与 bootstrap 互为交叉验证
    t_stat = d_mean / (sd / math.sqrt(n))
    try:
        from scipy import stats as _st
        p_val = float(2.0 * _st.t.sf(abs(t_stat), n - 1))
    except Exception:                                   # 无 scipy 时用正态近似
        p_val = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
    prim["paired_t"] = {"t": float(t_stat), "df": n - 1, "p_two_sided": p_val}

    # ---- 敏感性口径（同 test 视角）
    e_keys = list(per_energy)
    cnn_e = np.array([per_energy[e]["cnn_row16_test_mean"] for e in e_keys])
    lin_e = np.array([per_energy[e]["linear_row16_test_mean"] for e in e_keys])
    sens = {
        "per_energy_equal_weight_ratio_of_means": float(cnn_e.mean() / lin_e.mean()),
        "mean_of_per_energy_ratios": float(np.mean(cnn_e / lin_e)),
        "pooled_rms_ratio": float(np.sqrt(np.mean(cnn_e ** 2)) / np.sqrt(np.mean(lin_e ** 2))),
        "note": "同一批 180 个 test 视角的三种嵌套聚合；比值定义不同故数值不同",
    }

    # ---- 全视角口径（＝R59 表的聚合，仅供追溯，不作主指标）
    cnn_e_all = np.array([per_energy[e]["cnn_row16_all_views_mean"] for e in e_keys])
    lin_e_all = np.array([per_energy[e]["linear_row16_all_views_mean"] for e in e_keys])
    allview = {
        "pooled_rms_ratio": float(np.sqrt(np.mean(cnn_e_all ** 2))
                                  / np.sqrt(np.mean(lin_e_all ** 2))),
        "per_energy_equal_weight_ratio_of_means": float(cnn_e_all.mean() / lin_e_all.mean()),
        "mean_of_per_energy_ratios": float(np.mean(cnn_e_all / lin_e_all)),
        "note": "R59 表用的聚合（含 train/val 视角，360 视角/能量）；仅用于追溯已发布数字",
    }

    out = {
        "_meta": {
            "batch": "3_3_flaw_2_2", "defect_rod": "center",
            "row": "Row 16 (RECON_ROW)", "unmasked": True, "n_valid_px": 1200,
            "cnn_seed_aggregation": "per-view RMSE 的 3 种子算术平均",
            "linear_formula": "0.5*(sino[a-1]+sino[a+1]) - sino[a]",
            "seeds": [d.name for d in seed_dirs(ENERGIES[0])],
            "order_check": "corr(per_view, |sin θ|) 与 image_metrics.json 逐条一致",
        },
        "primary": prim,
        "sensitivity_test_only": sens,
        "all_views_legacy": allview,
        "per_energy": per_energy,
        "order_checks": corr_checks,
        "labels": {
            "primary": f"equal-weight paired, 180 (energy,view) observations, ratio={ratio:.4f}",
            "sensitivity": [
                f"per-energy equal-weight ratio-of-means = {sens['per_energy_equal_weight_ratio_of_means']:.4f}",
                f"mean of per-energy ratios = {sens['mean_of_per_energy_ratios']:.4f}",
            ],
            "diagnostic": f"pooled RMS ratio = {sens['pooled_rms_ratio']:.4f} (variance diagnostic, not primary)",
        },
    }
    p = RUNS / "_flaw22_row16_paired.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ PRIMARY (test-only, n=180) ================")
    print(f"ratio = {ratio:.4f}   95% CI [{prim['ratio_ci95'][0]:.4f}, {prim['ratio_ci95'][1]:.4f}]")
    print(f"Δ = {d_mean:+.6f}  CI [{prim['delta_ci95'][0]:+.6f}, {prim['delta_ci95'][1]:+.6f}]")
    print(f"improvement = {imp:+.2f}%  CI [{prim['improvement_ci95'][0]:+.2f}, "
          f"{prim['improvement_ci95'][1]:+.2f}]  d_z = {dz:+.3f}")
    print(f"paired t = {t_stat:+.3f} (df={n-1})  p = {p_val:.4g}")
    print("\nper-energy (test-only, 45 views each):")
    for e in e_keys:
        d = per_energy[e]
        print(f"  {e:>4} MeV  CNN={d['cnn_row16_test_mean']:.6f}  "
              f"lin={d['linear_row16_test_mean']:.6f}  ratio={d['ratio_test']:.4f}  "
              f"(all-view {d['ratio_all_views']:.4f})")
    print(f"\nsensitivity(test): ratio-of-means={sens['per_energy_equal_weight_ratio_of_means']:.4f} "
          f"mean-of-ratios={sens['mean_of_per_energy_ratios']:.4f} "
          f"pooledRMS={sens['pooled_rms_ratio']:.4f}")
    print(f"legacy(all-view): pooledRMS={allview['pooled_rms_ratio']:.4f} "
          f"ratio-of-means={allview['per_energy_equal_weight_ratio_of_means']:.4f} "
          f"mean-of-ratios={allview['mean_of_per_energy_ratios']:.4f}")
    print(f"[OK] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
