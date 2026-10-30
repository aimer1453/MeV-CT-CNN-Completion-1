#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/p01_mf_methods.py —— 缺陷信号的"三源"轨迹匹配滤波对比（绝杀实验）

问题
----
之前报出的 t = −7.90 是**数据级**量：`L_flaw − L_uniform` 两张实测正弦图之差，
完全不经过模型。审稿人会问：那 CNN 到底有没有用？

本脚本对**同一个匹配滤波算子**跑三种数据来源，得到可直接对比的三个 t：

  1. `measured`：`L_flaw` 实测（360 个视角全实测）→ 应复现 t = −7.90（自检项）
  2. `linear`  ：线性补全 `½[L(θ−1)+L(θ+1)]`（部署合法，只用实测邻居）
  3. `cnn`     ：CNN 补全（best.pt，每个视角用其相反相位的 θ±1/θ±3 作输入）

三者都减去**同一张** `L_uniform`（同 fill），再用同一套 D4 零假设：
把缺陷轨迹人为旋转 90/180/270° 各算一次，三次的平均即"伪轨迹基线"，
offset 0 相对该基线的偏离 = 缺陷信号；`t = contrast / sem`。

判读
----
- t_cnn ≈ t_measured  → 补全把缺陷特征基本保住了；
- t_cnn « t_measured  → 补全把缺陷抹掉了一部分；
- t_cnn 与 t_linear 的差 = CNN 相对线性插值在**缺陷可检出性**上的净贡献。
  注意：这不是"补全创造了缺陷"，而是"补全是否破坏了已有信息"。

输出
----
  runs/<flaw_run>_flaw_step3/mf_methods.json
  runs/<flaw_run>_flaw_step3/mf_methods_curves.npz   （三条 ΔL(θ) 曲线，供出图）

用法
----
  conda run -n ct_full --no-capture-output python scripts/p01_mf_methods.py
  conda run -n ct_full --no-capture-output python scripts/p01_mf_methods.py \
      --flaw-run-id 20261023_2327_cnn_bs8_lr1e-4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import baseline_recon as br                    # noqa: E402
import p01_flaw_step3_eval as S                # noqa: E402
import p01_reviewer_p0 as P0                   # noqa: E402

FLAW_RUN = "20261023_2327_cnn_bs8_lr1e-4"
BAND_PX = 60.0          # 轨迹带宽（与 p01_flaw_step3_eval 一致）
TRAJ = S.TRAJ           # {"amp":367.1,"offset":601.9,"phase_deg":224.7}
NAN = float("nan")


# ------------------------------------------------------------------ 匹配滤波
def traj_col(angles: np.ndarray, off_deg: float) -> np.ndarray:
    return TRAJ["offset"] + TRAJ["amp"] * np.cos(
        np.radians(angles + TRAJ["phase_deg"] + off_deg))


def mf_one(d: np.ndarray, band_px: float = BAND_PX) -> dict:
    """对一张差分正弦图 d(360,1200) 做 D4 对照的轨迹匹配滤波。"""
    cols = np.arange(d.shape[1], dtype=np.float64)
    angles = np.arange(d.shape[0], dtype=np.float64)
    out: dict = {}
    curves: dict[int, np.ndarray] = {}
    for off in (0, 90, 180, 270):
        traj = traj_col(angles, float(off))
        dl = np.empty(len(angles))
        for a in range(len(angles)):
            m = np.abs(cols - traj[a]) <= band_px
            dl[a] = d[a][m].mean() - d[a][~m].mean()
        curves[off] = dl
        mu = float(dl.mean())
        sem = float(dl.std(ddof=1) / math.sqrt(len(dl)))
        out[f"offset_{off}deg"] = {"mean_dlog": mu, "sem": sem,
                                   "t_stat": mu / sem if sem > 0 else NAN,
                                   "n_angles": len(angles)}
    base = [out[f"offset_{o}deg"]["mean_dlog"] for o in (90, 180, 270)]
    base_sem = [out[f"offset_{o}deg"]["sem"] for o in (90, 180, 270)]
    b0 = float(np.mean(base))
    b0_sem = float(np.sqrt(sum(s * s for s in base_sem)) / len(base))
    d0, s0 = out["offset_0deg"]["mean_dlog"], out["offset_0deg"]["sem"]
    contrast = d0 - b0
    csem = math.sqrt(s0 * s0 + b0_sem * b0_sem)
    out["d4_contrast"] = {
        "offset0_minus_pseudo_baseline": contrast,
        "sem": csem,
        "t_stat": contrast / csem if csem > 0 else NAN,
        "pseudo_baseline": b0,
        "pseudo_baseline_spread": float(np.ptp(base)),
        "offset0_mean": d0,
        "offset0_sem": s0,
    }
    return {"stats": out, "curves": curves}


def arc_band_stats(d: np.ndarray, peak_deg: float, half_deg: float = 18.0) -> dict:
    """只看缺陷可见弧（±half_deg）内的逐角 ΔL，看补全在"难的那一段"表现如何。"""
    angles = np.arange(d.shape[0], dtype=np.float64)
    dev = np.abs(((angles - peak_deg + 180.0) % 360.0) - 180.0)
    cols = np.arange(d.shape[1], dtype=np.float64)
    traj = traj_col(angles, 0.0)
    dl = np.empty(len(angles))
    for a in range(len(angles)):
        m = np.abs(cols - traj[a]) <= BAND_PX
        dl[a] = d[a][m].mean() - d[a][~m].mean()
    inside = dev <= half_deg
    outside = dev > half_deg
    return {
        "n_inside": int(inside.sum()),
        "dl_inside_mean": float(dl[inside].mean()),
        "dl_inside_sem": float(dl[inside].std(ddof=1) / math.sqrt(int(inside.sum()))),
        "dl_outside_mean": float(dl[outside].mean()),
        "dl_outside_sem": float(dl[outside].std(ddof=1) / math.sqrt(int(outside.sum()))),
        "inside_minus_outside": float(dl[inside].mean() - dl[outside].mean()),
        "half_width_deg": half_deg,
    }


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="三源轨迹匹配滤波对比")
    ap.add_argument("--flaw-run-id", default=FLAW_RUN)
    args = ap.parse_args(argv)
    run_id = args.flaw_run_id
    out_dir = ROOT / "runs" / f"{run_id}_flaw_step3"
    if not out_dir.is_dir():
        raise SystemExit(f"[FAIL] 缺少 {out_dir}")

    cfg = json.loads((br.RUNS_DIR / run_id / "config.json").read_text(encoding="utf-8"))
    # 数据表与数据根都从 run 自己反推（config.data.targets 只记表名）。
    # 2026-10-28 的坑：不反推就用模块默认的 sparse_targets_flaw.csv + 3_3_flaw，
    # 于是 measured/linear 两条线读的还是上一代数据，只有 norm 和 CNN 权重是新的。
    from data.csv_paths import resolve_csv
    S.FLAW_CSV = resolve_csv(cfg["data"]["targets"])
    _rows_probe = S.read_rows(S.FLAW_CSV)
    S.FLAW_ROOT = (br.ROOT / _rows_probe[0]["target_file"]).parent.parent.parent
    print(f"[数据] {S.FLAW_CSV.name}  root={S.FLAW_ROOT.relative_to(br.ROOT)}", flush=True)
    norm = cfg["norm"]
    mean, std = float(norm["norm_mean"]), float(norm["norm_std"])
    fill = float(norm["zero_log_fill"])
    print(f"run={run_id}  norm mean={mean} std={std} fill={fill}", flush=True)

    # ---- 实测：缺陷侧与无缺陷侧，同 fill、同角度序
    rows_f = S.read_rows(S.FLAW_CSV)
    S.check_angles(rows_f, "flaw")
    # 能量也跟着 run 走。两个表的 energy_MeV token 不同形：缺陷表写 `6.0`、
    # 均匀表写 `6`，用字符串相等会静默选出 0 行（后续 build_sinogram 直接崩）或选错档。
    S.ENERGY = rows_f[0]["energy_MeV"]
    _e = float(S.ENERGY)
    print(f"[能量] {S.ENERGY} MeV（按数值匹配均匀表，token 可能是 `3` 而非 `3.0`）", flush=True)
    sino_f, theta_f, _ = S.build_sinogram(rows_f, S.FLAW_ROOT, fill)
    rows_u = [r for r in S.read_rows(br.SPARSE_TARGETS)
              if abs(float(r["energy_MeV"]) - _e) < 1e-9]
    if len(rows_u) != 360:
        raise SystemExit(f"[FAIL] 均匀表在 {_e} MeV 上取到 {len(rows_u)} 行，应为 360")
    rows_u.sort(key=lambda r: float(r["target_angle_deg"]))
    S.check_angles(rows_u, "uniform")
    sino_u, theta_u, _ = S.build_sinogram(rows_u, ROOT / "3_3", fill)
    a_f = np.round(theta_f).astype(np.int64)
    a_u = np.round(theta_u).astype(np.int64)
    if not np.array_equal(a_f, a_u) or not np.array_equal(a_u, np.arange(360)):
        raise SystemExit("两侧角度序不一致或不是 0..359，不可逐角相减")
    print(f"正弦图: flaw {sino_f.shape}  uniform {sino_u.shape}", flush=True)

    d_meas = sino_f - sino_u

    # ---- 线性补全：½[L(θ−1)+L(θ+1)]（只用实测邻居，部署合法）
    idx = np.arange(360)
    L_lin = 0.5 * (sino_f[(idx - 1) % 360] + sino_f[(idx + 1) % 360])
    d_lin = L_lin - sino_u

    # ---- CNN 补全：每个视角用其相反相位的 θ±1/θ±3 作输入，全 360 视角覆盖
    print("CNN 推理（360 个视角各一次）...", flush=True)
    pred, _ = P0.predict_all(rows_f, br.RUNS_DIR / run_id, mean, std, fill)
    L_cnn = pred[:, 0, br.RECON_ROW, :].astype(np.float64)
    if L_cnn.shape != sino_u.shape:
        raise SystemExit(f"CNN 形状 {L_cnn.shape} != {sino_u.shape}")
    d_cnn = L_cnn - sino_u

    # ---- 三源匹配滤波
    res: dict = {"run_id": run_id, "energy_MeV": S.ENERGY, "band_halfwidth_px": BAND_PX,
                 "trajectory": TRAJ, "fill_flaw": fill,
                 "note": ("三个来源减同一张 L_uniform、用同一 D4 零假设；"
                          "measured 不含模型，是自检项")}
    curves: dict[str, np.ndarray] = {}
    curves_all: dict[str, dict[int, np.ndarray]] = {}
    srcs = (("measured", d_meas), ("linear", d_lin), ("cnn", d_cnn))
    for name, d in srcs:
        r = mf_one(d)
        res[name] = r["stats"]
        curves[name] = r["curves"][0]
        curves_all[name] = r["curves"]
        peak = float(np.argmax(curves[name]))
        res[name]["arc_band"] = arc_band_stats(d, peak)
        res[name]["peak_angle_deg"] = peak

    # D4 校正后的逐角信号 c(θ) = ΔL_0(θ) − mean[ΔL_90, ΔL_180, ΔL_270](θ)
    # 三条伪轨迹用的是同一套角度，所以不同方法之间天然配对
    for name, _ in srcs:
        cc = curves_all[name]
        c_curve = cc[0] - np.mean([cc[90], cc[180], cc[270]], axis=0)
        res[name]["d4_curve"] = {"mean": float(c_curve.mean()),
                                 "sem": float(c_curve.std(ddof=1) / math.sqrt(len(c_curve)))}
        curves_all[name]["d4"] = c_curve

    # ---- 相对 measured 的信号保留率
    c_meas = res["measured"]["d4_contrast"]["offset0_minus_pseudo_baseline"]
    t_meas = res["measured"]["d4_contrast"]["t_stat"]
    for name in ("linear", "cnn"):
        c = res[name]["d4_contrast"]["offset0_minus_pseudo_baseline"]
        res[name]["contrast_retained_vs_measured"] = c / c_meas
        res[name]["t_ratio_vs_measured"] = res[name]["d4_contrast"]["t_stat"] / t_meas

    # ---- 配对检验：同一组 360 个视角上的 ΔL(θ) 逐角差（三者共用同一 gt，天然配对）
    def paired(a: np.ndarray, b: np.ndarray, tag: str) -> dict:
        diff = a - b
        n = len(diff)
        mu = float(diff.mean())
        sem = float(diff.std(ddof=1) / math.sqrt(n))
        return {"tag": tag, "mean_diff": mu, "sem": sem,
                "t_stat": mu / sem if sem > 0 else NAN, "n_angles": n}

    cm, cl, cc = curves["measured"], curves["linear"], curves["cnn"]
    dm = curves_all["measured"]["d4"]
    dl4 = curves_all["linear"]["d4"]
    dc4 = curves_all["cnn"]["d4"]
    res["paired_tests"] = {
        "linear_minus_measured": paired(cl, cm, "ΔL: linear - measured"),
        "cnn_minus_measured": paired(cc, cm, "ΔL: cnn - measured"),
        "cnn_minus_linear": paired(cc, cl, "ΔL: cnn - linear"),
        "d4: linear_minus_measured": paired(dl4, dm, "D4: linear - measured"),
        "d4: cnn_minus_measured": paired(dc4, dm, "D4: cnn - measured"),
        "d4: cnn_minus_linear": paired(dc4, dl4, "D4: cnn - linear"),
    }
    # ΔL(θ) 曲线对 measured 的保真度（逐角 RMS 偏差，越小越"没改坏"）
    for name in ("linear", "cnn"):
        dev = curves[name] - cm
        res[name]["curve_rms_vs_measured"] = float(np.sqrt(np.mean(dev ** 2)))
        res[name]["curve_maxabs_vs_measured"] = float(np.max(np.abs(dev)))

    # ---- 打印
    print("\n" + "=" * 96, flush=True)
    print(f"{'source':<10} {'offset0 ΔL':>12} {'pseudo base':>12} {'contrast':>10} "
          f"{'sem':>8} {'t':>8} {'kept':>7} {'arc in/out':>12}", flush=True)
    print("-" * 96, flush=True)
    for name, _ in srcs:
        c = res[name]["d4_contrast"]
        kept = res[name].get("contrast_retained_vs_measured", 1.0)
        ab = res[name]["arc_band"]
        print(f"{name:<10} {c['offset0_mean']:>12.5f} {c['pseudo_baseline']:>12.5f} "
              f"{c['offset0_minus_pseudo_baseline']:>10.5f} {c['sem']:>8.5f} "
              f"{c['t_stat']:>8.2f} {kept:>6.1%} "
              f"{ab['dl_inside_mean']:>+6.4f}/{ab['dl_outside_mean']:>+6.4f}", flush=True)
    print("=" * 96, flush=True)
    print(f"缺陷可见弧 ±{res['measured']['arc_band']['half_width_deg']:.0f}°，"
          f"弧内 {res['measured']['arc_band']['n_inside']} 个视角", flush=True)
    print("\n弧内−弧外（局部对比度，更抗全局尺度漂移）：", flush=True)
    for name, _ in srcs:
        print(f"  {name:<10} {res[name]['arc_band']['inside_minus_outside']:+.5f}", flush=True)
    print("\n配对检验（逐角，n=360）：", flush=True)
    for k, v in res["paired_tests"].items():
        print(f"  {v['tag']:<28} {v['mean_diff']:+.5f} ± {v['sem']:.5f}  "
              f"t = {v['t_stat']:+.2f}", flush=True)
    print("\nD4 校正曲线 c(θ) 的均值（= 表里的 contrast）：", flush=True)
    for name, _ in srcs:
        d4 = res[name]["d4_curve"]
        print(f"  {name:<10} {d4['mean']:+.5f} ± {d4['sem']:.5f}", flush=True)
    print("\nΔL(θ) 对 measured 的保真度：", flush=True)
    for name in ("linear", "cnn"):
        print(f"  {name:<10} RMS {res[name]['curve_rms_vs_measured']:.5f}  "
              f"max|Δ| {res[name]['curve_maxabs_vs_measured']:.5f}", flush=True)

    (out_dir / "mf_methods.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    arrays: dict[str, np.ndarray] = {}
    for name, _ in srcs:
        for off, arr in curves_all[name].items():
            arrays[f"{name}_off{off}"] = arr
        arrays[f"{name}_d4"] = curves_all[name]["d4"]
    np.savez(out_dir / "mf_methods_curves.npz", **arrays)
    print(f"\n输出: {out_dir / 'mf_methods.json'}", flush=True)
    print(f"      {out_dir / 'mf_methods_curves.npz'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
