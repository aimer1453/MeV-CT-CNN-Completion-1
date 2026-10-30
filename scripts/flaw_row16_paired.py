#!/usr/bin/env python
"""corner / edge 缺陷位置的 Row-16 逐视角配对主指标 —— 与 center (`flaw22`) 及 §5.6 同口径。

为什么需要这个脚本
------------------
`_flaw22_row16_paired.json` 只覆盖中心棒批（`3_3_flaw_2_2`）。corner（`3_3_flaw_1_1`）与
edge（`3_3_flaw_1_2`）当时的 eval 只存了 **all-360-view** 的 Row-16 聚合（Table 8 的
1.150 即 3.0 MeV 的 all-360 ratio），没存 test-only 配对。本脚本用与 center 完全相同的
管线补齐 test-only，**不重跑训练**：

* 只取 **test 视角**：4 能量 × 45 = **180 个 (energy, view) 配对观测**
* 每视角 Row-16 RMSE = sqrt(mean(误差² over 1200 列))，**unmasked**（n_valid = 1200，
  与 §5.6 的 per_sample_test 一致；已用 Table 8 的 all-360 值反证 unmasked 才是原文口径）
* CNN 每视角 = **3 个种子 per-view RMSE 的算术平均**
* linear = 0.5*(sino[a−1] + sino[a+1]) − sino[a]，与 `recon_all()` 逐字同式
* 配对 bootstrap 10000 次、seed 20261022（与 `dose_bootstrap.py` 同参）

符号约定（与 center 一致）
------------------------
Δ = CNN − linear（**正 = CNN 更差**）；ratio = mean(CNN)/mean(linear)
improvement% = −Δ/mean(linear) × 100（**正 = CNN 更好**）

自检（fail-fast）
-----------------
* 全 360 视角的 Row-16 均值必须复现已公布的表值（flaw12 = Table 8 / tab:flawedge；
  flaw22 = R59 的 per-energy 值）；容差 1.5e-4（表值只有 4 位小数）
* 每档必须恰好 3 个种子目录、45 个 test 视角、360 行 CSV

产出：runs/_<batch>_row16_paired.json
用法：python scripts/flaw_row16_paired.py --batch flaw12
"""
from __future__ import annotations

import argparse
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
TOL_TAB = 1.5e-4          # 表值 4 位小数 → 半位 5e-5，留 3 倍余量

# 已公布的 all-360-view Row-16 值（CNN, linear），用于自检
PUBLISHED = {
    # Table 8 / tab:flawedge（edge rod, edge2, r=260 px, az=180 deg）
    "flaw12": {"1.5": (0.1246, 0.1258), "3.0": (0.0692, 0.0602),
               "4.5": (0.0591, 0.0537), "6.0": (0.0586, 0.0540)},
    # R59 表（center rod）
    "flaw22": {"1.5": (0.122467, 0.125762), "3.0": (0.069223, 0.059673),
               "4.5": (0.058827, 0.053077), "6.0": (0.056976, 0.053607)},
    # corner 批（3_3_flaw_1_1）没有已公布的 Row-16 表值 → 不设自检
    "flaw11": {},
}

LABEL = {"flaw11": "corner rod (3_3_flaw_1_1)",
         "flaw12": "edge rod (3_3_flaw_1_2)",
         "flaw22": "centre rod (3_3_flaw_2_2)"}


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def build_sino(rows: list[dict], fill: float) -> tuple[np.ndarray, np.ndarray]:
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


def err_row_index(a: int) -> tuple[str, int]:
    """角度 → (误差图相位, 行号)。errs[in_phase] 存**缺失相位**（即预测视角）的误差：
    phase A = 偶角 → errs['B']；phase B = 奇角 → errs['A']。"""
    if a % 2 == 1:
        return "A", (a - 1) // 2
    return "B", a // 2


def ci(mb: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))


def run(batch: str, allow_mismatch: bool = False, out: Path | None = None) -> dict:
    per_energy: dict[str, dict] = {}
    pooled_cnn, pooled_lin = [], []
    checks: list[dict] = []

    for E in ENERGIES:
        csv_p = CSV_DIR / f"sparse_targets_{batch}_{E}.csv"
        if not csv_p.exists():
            raise SystemExit(f"[FATAL] 缺 CSV {csv_p}")
        rows = list(csv.DictReader(open(csv_p, newline="", encoding="utf-8")))
        if len(rows) != 360:
            raise SystemExit(f"[FATAL] {csv_p.name} 行数 {len(rows)} != 360")

        ds = sorted(d for d in RUNS.glob(f"*_{batch}_{E}_flaw_step3") if d.is_dir())
        if len(ds) != 3:
            raise SystemExit(f"[FATAL] {batch} {E} MeV 期望 3 个种子目录，实得 {len(ds)}")
        fills = {float(json.loads((d / "config.json").read_text(encoding="utf-8"))
                       ["norm"]["zero_log_fill"]) for d in ds}
        if len(fills) != 1:
            raise SystemExit(f"[FATAL] {batch} {E} MeV 三种子 fill 不一致: {fills}")
        fill = fills.pop()

        sino, theta = build_sino(rows, fill)
        idx = {int(round(float(t))): i for i, t in enumerate(theta)}
        if len(idx) != 360:
            raise SystemExit(f"[FATAL] theta 非 360 唯一角: {len(idx)}")

        # linear：全 360 视角逐视角 Row-16 误差（test 子集后面切）
        lin_pv = np.empty(360, dtype=np.float64)
        for a in range(360):
            pred = 0.5 * (sino[idx[(a - 1) % 360]] + sino[idx[(a + 1) % 360]]) - sino[idx[a]]
            lin_pv[a] = rms(pred)

        # CNN：3 种子 per-view RMSE 的算术平均
        stack = []
        for d in ds:
            eA = np.load(d / "proj_error_map_A.npy").astype(np.float64)
            eB = np.load(d / "proj_error_map_B.npy").astype(np.float64)
            if eA.shape != (180, br.OUTPUT_SIZE) or eB.shape != (180, br.OUTPUT_SIZE):
                raise SystemExit(f"[FATAL] {d.name} 误差图形状异常 {eA.shape} {eB.shape}")
            arr = np.empty(360, dtype=np.float64)
            for a in range(360):
                ph, i = err_row_index(a)
                arr[a] = rms((eA if ph == "A" else eB)[i])
            stack.append(arr)
        cnn_pv = np.mean(np.stack(stack, axis=0), axis=0)

        # 自检：全 360 视角均值复现已公布表值
        ref = PUBLISHED.get(batch, {}).get(E)
        cnn_all, lin_all = float(cnn_pv.mean()), float(lin_pv.mean())
        if ref is not None:
            d_c, d_l = abs(cnn_all - ref[0]), abs(lin_all - ref[1])
            ok = max(d_c, d_l) <= TOL_TAB
            rec = {"energy": E, "cnn_all": cnn_all, "linear_all": lin_all,
                   "published": list(ref), "max_abs_diff": max(d_c, d_l), "reproduced": ok}
            checks.append(rec)
            if not ok:
                msg = (f"[WARN] {batch} {E} MeV 未复现表值：CNN {cnn_all:.6f} vs {ref[0]:.4f}"
                       f"（Δ{d_c:+.2e}）；linear {lin_all:.6f} vs {ref[1]:.4f}（Δ{d_l:+.2e}）")
                n_bad = sum(1 for c in checks if not c["reproduced"])
                if n_bad * 2 > len(checks) and len(checks) >= 2:
                    raise SystemExit(msg + " —— 过半能量不吻合，按系统性错误中止")
                if not allow_mismatch:
                    raise SystemExit(msg + " —— 严格执行；如需记录后继续请加 --allow-mismatch")
                print(msg + "  [recorded as provenance gap, continuing]", flush=True)

        test_ang = sorted({int(float(r["target_angle_deg"]))
                           for r in rows if r["split"] == "test"})
        if len(test_ang) != 45:
            raise SystemExit(f"[FATAL] {batch} {E} MeV test 视角 {len(test_ang)} != 45")
        ti = np.array(test_ang, dtype=int)
        c_t, l_t = cnn_pv[ti], lin_pv[ti]
        pooled_cnn.append(c_t)
        pooled_lin.append(l_t)

        dd = c_t - l_t
        d_m, d_s = float(dd.mean()), float(dd.std(ddof=1))
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
            "published_all_views": list(ref) if ref else None,
            "test_angles": test_ang,
        }
        print(f"[{batch} {E} MeV] test(45): CNN={c_t.mean():.6f} lin={l_t.mean():.6f} "
              f"ratio={c_t.mean()/l_t.mean():.4f} (CI {r_ci[0]:.4f}-{r_ci[1]:.4f}) "
              f"p={p_e:.3g} | all-360 ratio={cnn_all/lin_all:.4f}", flush=True)

    # ---- 池化主指标：180 个 (energy, view) 等权配对
    cnn = np.concatenate(pooled_cnn)
    lin = np.concatenate(pooled_lin)
    dd = cnn - lin
    n = len(dd)
    mean_cnn, mean_lin = float(cnn.mean()), float(lin.mean())
    d_mean = float(dd.mean())
    ratio = mean_cnn / mean_lin
    imp = -d_mean / mean_lin * 100.0
    t_val = d_mean / (dd.std(ddof=1) / math.sqrt(n))
    try:
        from scipy import stats as _st
        p_val = float(2.0 * _st.t.sf(abs(t_val), n - 1))
    except Exception:
        p_val = float(math.erfc(abs(t_val) / math.sqrt(2.0)))
    dz = d_mean / dd.std(ddof=1)

    rb = np.random.default_rng(SEED)
    bi = rb.integers(0, n, size=(N_BOOT, n))
    ratio_b = cnn[bi].mean(axis=1) / lin[bi].mean(axis=1)
    delta_b = dd[bi].mean(axis=1)
    imp_b = -delta_b / lin[bi].mean(axis=1) * 100.0

    primary = {
        "definition": "equal-weight paired over 180 (energy, view) test observations",
        "n": n,
        "mean_cnn": mean_cnn, "mean_linear": mean_lin,
        "delta_cnn_minus_linear": d_mean,
        "delta_ci95": list(ci(delta_b)),
        "ratio_cnn_over_linear": ratio,
        "ratio_ci95": list(ci(ratio_b)),
        "improvement_pct": imp,
        "improvement_ci95": list(ci(imp_b)),
        "cohens_dz": float(dz),
        "paired_t": {"t": float(t_val), "df": n - 1, "p_two_sided": p_val},
        "n_boot": N_BOOT, "seed": SEED,
    }

    # 敏感性：逐能量比值的算术平均 / pooled RMS
    ratios = [per_energy[E]["ratio_test"] for E in ENERGIES]
    pooled_rms = (math.sqrt(np.mean([per_energy[E]["cnn_row16_test_mean"] ** 2 for E in ENERGIES]))
                  / math.sqrt(np.mean([per_energy[E]["linear_row16_test_mean"] ** 2 for E in ENERGIES])))

    result = {
        "batch": batch,
        "label": LABEL.get(batch, batch),
        "primary": primary,
        "sensitivity": {
            "mean_of_per_energy_ratios": float(np.mean(ratios)),
            "per_energy_ratios": dict(zip(ENERGIES, ratios)),
        },
        "diagnostic": {
            "pooled_rms_ratio": float(pooled_rms),
            "note": "variance diagnostic, not primary",
        },
        "per_energy": per_energy,
        "self_check_vs_published": {
            "checked": bool(checks),
            "tolerance": TOL_TAB,
            "n_reproduced": sum(1 for c in checks if c["reproduced"]),
            "n_checked": len(checks),
            "mismatched_energies": [c["energy"] for c in checks if not c["reproduced"]],
            "rows": checks,
            "note": ("all-360-view Row-16 means reproduce the published table"
                     if checks else "no published all-360 Row-16 row for this batch"),
        },
        "caliber": {
            "views": "test split only, 45 per energy x 4 energies = 180 observations",
            "row16_rmse": "unmasked, sqrt(mean(err^2)) over 1200 columns",
            "cnn": "arithmetic mean of 3 seeds' per-view RMSE",
            "linear": "0.5*(sino[a-1]+sino[a+1]) - sino[a]",
            "bootstrap": f"{N_BOOT} resamples, seed {SEED}",
        },
    }
    out_p = out or (RUNS / f"_{batch}_row16_paired.json")
    out_p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== {batch} primary (n={n}) ===")
    print(f"  ratio = {ratio:.4f}  CI [{primary['ratio_ci95'][0]:.4f}, {primary['ratio_ci95'][1]:.4f}]")
    print(f"  deficit = {imp:+.2f}%  CI [{primary['improvement_ci95'][0]:+.2f}, "
          f"{primary['improvement_ci95'][1]:+.2f}]  d_z = {dz:+.3f}")
    print(f"  paired t = {t_val:.3f} (df={n-1})  p = {p_val:.3g}")
    print(f"  sensitivity: mean of per-energy ratios = {np.mean(ratios):.4f} | "
          f"pooled RMS = {pooled_rms:.4f}")
    if checks:
        bad = [c["energy"] for c in checks if not c["reproduced"]]
        print(f"  self-check: {len(checks) - len(bad)}/{len(checks)} energies reproduce the "
              f"published table (max |diff| = {max(c['max_abs_diff'] for c in checks):.2e})"
              + (f"; NOT reproduced: {', '.join(bad)} MeV" if bad else ""))
    else:
        print("  self-check: no published all-360 Row-16 row for this batch")
    print(f"  -> {out_p}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, choices=sorted(PUBLISHED))
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="表值不复现时记录为 provenance gap 并继续（默认中止）")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出路径（默认 runs/_<batch>_row16_paired.json；校验时避免覆盖已交付产物）")
    a = ap.parse_args()
    run(a.batch, allow_mismatch=a.allow_mismatch, out=a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
