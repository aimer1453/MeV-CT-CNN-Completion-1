#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/p1_multiseed.py —— P1 多种子统计驱动脚本

目的
----
反击"单次运行无法证明稳定性"的审稿意见：对同一配置用不同随机种子重跑，
报告 row16 / all32 / 图像域 的 mean ± std 与 95% 置信区间。

口径（与论文正文完全一致，不得另立）
----------------------------------
- 均匀数据：`sparse_targets.csv` + `config.json`（全局 norm 0.7202/0.9249/5.50）
- 缺陷数据：`sparse_targets_flaw.csv` + `config_flaw.json`（缺陷 norm 0.5174/0.8145/5.10）
- 训练超参恒为：batch=8, lr=1e-4, epochs=100, warmup=50 step, AMP=off
- best.pt 按 val_rmse_valid_phys 保存（与所有已发布 run 同一规则）
- 只改 `seed` 一项，其余一律不动

输出
----
`runs/_reviewer_p0/multiseed_summary.json`
    {
      "uniform": {"20261022": {...}, "20261023": {...}, "20261024": {...}},
      "flaw":    {...},
      "stats":   {"uniform": {"row16": {"mean":..,"std":..,"ci95":..}, ...}, ...}
    }

用法
----
    python scripts/p1_multiseed.py                       # 跑全部缺的种子
    python scripts/p1_multiseed.py --seeds 20261023 20261024
    python scripts/p1_multiseed.py --domain flaw --seeds 20261022
    python scripts/p1_multiseed.py --aggregate-only      # 只汇总，不训练

已完成的种子会自动跳过（按 summary 里是否存在该条目判断），可安全重入。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "_reviewer_p0" / "multiseed_summary.json"
LOG_DIR = ROOT / ".workbuddy" / "tmp" / "multiseed"
PY = sys.executable

#: 已发布的基准 run（seed=20261022，100 epoch）。多种子统计把它作为第一颗种子纳入，
#: 不重跑，直接读它已有的产物。
BASE_UNIFORM = "20261024_0400_cnn_bs8_lr1e-4"
BASE_FLAW = "20261024_1558_cnn_bs8_lr1e-4"
ENERGIES = ("1.5", "3", "4.5", "6")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], logfile: Path) -> str:
    """跑子进程，全部输出落盘；返回 stdout 文本。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(logfile, "w", encoding="utf-8") as fh:
        fh.write(" ".join(cmd) + "\n\n")
    with open(logfile, "a", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"命令失败({proc.returncode}): {' '.join(cmd)}  日志: {logfile}")
    return logfile.read_text(encoding="utf-8", errors="replace")


def newest_run(tag: str) -> str:
    """取最新的 runs/<YYYYMMDD_HHMM>_cnn_bs8_lr1e-4 目录名。"""
    cands = [p for p in RUNS.iterdir()
             if p.is_dir() and re.fullmatch(r"\d{8}_\d{4}_cnn_bs8_lr1e-4", p.name)]
    if not cands:
        raise RuntimeError("找不到训练 run 目录")
    return max(cands, key=lambda p: p.stat().st_mtime).name


# --------------------------------------------------------------- 指标提取
def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def uniform_metrics(run_id: str, seed: int) -> dict:
    """均匀数据：投影域 row16/all32（CNN 与 linear）+ 图像域 rmse。

    linear 的 row16 由 `p01_table_proj_domain.py --out-suffix _s<seed>` 现读正弦图算出
    （与论文 Table 4 同源，不另立口径）；CNN 的 row16/all32 取自
    `<run>_cnn_recon/test_metrics.json` 的 `projection_domain_check`。
    """
    cnn = load(RUNS / f"{run_id}_cnn_recon" / "test_metrics.json")
    lin_img = load(RUNS / f"{run_id}_linear_interp_image" / "test_metrics.json")
    lin_proj = load(RUNS / f"{run_id}_linear_interp" / "test_metrics.json")
    pdc = cnn["projection_domain_check"]
    tab = RUNS / "_reviewer_p0" / f"table_proj_domain_s{seed}.json"

    out: dict = {"run_id": run_id, "image": {}, "projection": {}}
    out["image"]["cnn_rmse_valid"] = cnn["overall"]["rmse_valid"]
    out["image"]["linear_rmse_valid"] = lin_img["overall"]["rmse_valid"]
    out["image"]["cnn_psnr_common"] = cnn["overall"].get("psnr_common")
    out["image"]["linear_psnr_common"] = lin_img["overall"].get("psnr_common")
    out["image"]["cnn_ssim"] = cnn["overall"].get("ssim")
    out["image"]["linear_ssim"] = lin_img["overall"].get("ssim")

    for roi in ("row16", "all32"):
        nv = se = 0.0
        for e in ENERGIES:
            for ph, v in pdc[e]["by_input_phase"].items():
                if roi not in v:          # missing_phase 占位
                    continue
                t = v[roi]["by_split"]["test"]
                nv += t["n_valid_pixels"]
                se += t["rmse_valid"] ** 2 * t["n_valid_pixels"]
        out["projection"][f"cnn_{roi}"] = math.sqrt(se / nv)
        out["projection"][f"cnn_{roi}_n_valid_pixels"] = int(nv)
    out["projection"]["linear_all32"] = lin_proj["overall"]["rmse_valid"]

    if tab.is_file():
        t = load(tab)["overall"]
        out["projection"]["linear_row16"] = t["linear_row16"]["rmse_valid"]
        # 交叉校验：表格脚本重算的 CNN row16/all32 应与上面池化值一致（tol 1e-9）
        for roi in ("row16", "all32"):
            a = out["projection"][f"cnn_{roi}"]
            b = t[f"cnn_{roi}"]["rmse_valid"]
            if abs(a - b) > 1e-9:
                raise RuntimeError(
                    f"CNN {roi} 两处池化不一致: {a!r} vs 表格 {b!r}（run={run_id}）")
    else:
        out["projection"]["linear_row16"] = None
    return out


def _retained(d: dict) -> float | None:
    """匹配滤波保持率：兼容两种键名。

    - `p01_mf_methods.py`（旧）      -> `contrast_retained_vs_measured`（0~1 小数）
    - `p01_flaw_step3_eval.py`（新） -> `retained_pct`（百分数）
    """
    if "retained_pct" in d:
        return float(d["retained_pct"])
    if "contrast_retained_vs_measured" in d:
        return float(d["contrast_retained_vs_measured"]) * 100.0
    return None


def _d4(d: dict) -> float | None:
    """D4 对照（offset0 − 伪轨迹基线）。键名兼容：
    `p01_mf_methods.py` 用 `offset0_minus_pseudo_baseline`。"""
    v = d.get("d4_contrast") or {}
    if "offset0_minus_pseudo_baseline" in v:
        return float(v["offset0_minus_pseudo_baseline"])
    if "contrast" in v:
        return float(v["contrast"])
    return None


def flaw_metrics(run_id: str) -> dict:
    """缺陷数据：投影域 row16/all32 + 图像域 + 匹配滤波保持率 + CNR。"""
    im = load(RUNS / f"{run_id}_flaw_step3" / "image_metrics.json")
    mf = load(RUNS / f"{run_id}_flaw_step3" / "mf_methods.json")
    tm = load(RUNS / run_id / "test_metrics.json")

    out: dict = {"run_id": run_id, "image": {}, "projection": {}, "matched_filter": {},
                 "cnr": {}}
    pm = im["per_method"]
    out["image"]["cnn_rmse_valid"] = pm["cnn_completed"]["rmse_valid"]
    out["image"]["linear_rmse_valid"] = pm["linear_interp"]["rmse_valid"]
    out["image"]["fbp_rmse_valid"] = pm["fbp_sparse"]["rmse_valid"]
    out["image"]["sart_rmse_valid"] = pm["sart_sparse"]["rmse_valid"]
    out["image"]["cnn_ssim"] = pm["cnn_completed"]["ssim"]
    out["image"]["linear_ssim"] = pm["linear_interp"]["ssim"]

    pd = im["projection_domain"]
    out["projection"]["cnn_all32_mean"] = pd.get("rmse_valid_mean")
    out["projection"]["domain_mismatch"] = pd.get("domain_mismatch", False)
    r16 = pd["per_view_row16"]
    n_tot = sum(v["n_views"] for v in r16.values())
    out["projection"]["cnn_row16_mean"] = sum(
        v["rmse_row16_mean"] * v["n_views"] for v in r16.values()) / n_tot
    out["projection"]["n_test_views"] = n_tot
    out["projection"]["trainpy_test_rmse_valid"] = tm["overall"]["rmse_valid"]

    out["matched_filter"]["cnn_retained_pct"] = _retained(mf["cnn"])
    out["matched_filter"]["linear_retained_pct"] = _retained(mf["linear"])
    out["matched_filter"]["cnn_d4_contrast"] = _d4(mf["cnn"])
    out["matched_filter"]["linear_d4_contrast"] = _d4(mf["linear"])
    out["matched_filter"]["cnn_curve_rms"] = mf["cnn"]["curve_rms_vs_measured"]
    out["matched_filter"]["linear_curve_rms"] = mf["linear"]["curve_rms_vs_measured"]
    pt = mf.get("paired_tests", {})
    for k, v in pt.items():
        if isinstance(v, dict) and "t_stat" in v:
            out["matched_filter"][f"t_{k.replace(': ', '_').replace(' ', '_')}"] = v["t_stat"]

    for meth, v in im["defect_cnr"].items():
        if isinstance(v, dict) and v.get("status") == "ok":
            out["cnr"][meth] = v["in_image"].get("cnr")
    return out


# --------------------------------------------------------------- 统计
def stats_of(vals: list[float]) -> dict:
    n = len(vals)
    mu = sum(vals) / n
    if n < 2:
        return {"n": n, "mean": mu, "std": float("nan"), "ci95_halfwidth": float("nan")}
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    # t 分布 95% 双侧临界值（n=3 → 4.303, n=2 → 12.706）
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 2.776)
    return {"n": n, "mean": mu, "std": sd,
            "ci95_halfwidth": tcrit * sd / math.sqrt(n),
            "t_critical": tcrit,
            "min": min(vals), "max": max(vals)}


def aggregate(summary: dict) -> dict:
    out: dict = {}
    for domain in ("uniform", "flaw"):
        runs = summary.get(domain, {})
        if len(runs) < 2:
            continue
        seeds = sorted(runs)
        st: dict = {}
        # 均匀：row16 / all32 / 图像域；缺陷：row16 / all32 / 图像域
        keys = (["cnn_row16", "cnn_all32", "linear_row16", "linear_all32",
                 "cnn_rmse_valid", "linear_rmse_valid", "cnn_psnr_common",
                 "linear_psnr_common", "cnn_ssim", "linear_ssim"]
                if domain == "uniform" else
                ["cnn_row16_mean", "cnn_all32_mean", "cnn_rmse_valid",
                 "linear_rmse_valid", "cnn_ssim", "linear_ssim",
                 "cnn_retained_pct", "linear_retained_pct",
                 "cnn_d4_contrast", "linear_d4_contrast",
                 "cnn_curve_rms", "linear_curve_rms"])
        for k in keys:
            vals = []
            for s in seeds:
                d = runs[s]
                if domain == "uniform":
                    if (k.endswith("_rmse_valid") or k.endswith("_psnr_common")
                            or k.endswith("_ssim")):
                        sub, kk = d["image"], k
                    else:
                        sub, kk = d["projection"], k
                else:
                    if "retained" in k or "d4_contrast" in k or "curve_rms" in k:
                        sub, kk = d["matched_filter"], k
                    elif k.endswith("_mean"):
                        sub, kk = d["projection"], k
                    elif k.endswith("_rmse_valid") or k.endswith("_ssim"):
                        sub, kk = d["image"], k
                    else:
                        continue
                if kk in sub and sub[kk] is not None and math.isfinite(float(sub[kk])):
                    vals.append(float(sub[kk]))
            if len(vals) >= 2:
                st[k] = {**stats_of(vals), "values": vals, "seeds": seeds}
        out[domain] = st
    return out


# --------------------------------------------------------------- 主流程
def ensure_base_uniform() -> None:
    """为 seed=20261022 的已发布 run 补齐多种子统计所需的两份产物。

    - `runs/<BASE_UNIFORM>_linear_interp/`：投影域线性插值（baseline_linear.py）
    - `runs/_reviewer_p0/table_proj_domain_s20261022.json`：linear row16（现读正弦图）
    两者在 100ep 定稿时都还没生成，缺了会让第一颗种子无法纳入统计。
    """
    lin = RUNS / f"{BASE_UNIFORM}_linear_interp" / "test_metrics.json"
    if not lin.is_file():
        log(f"  补跑 baseline_linear.py（{BASE_UNIFORM}）...")
        run([PY, "-u", "scripts/baseline_linear.py", "--cnn-run-id", BASE_UNIFORM],
            LOG_DIR / "baseline_linear_base.log")
    tab = RUNS / "_reviewer_p0" / "table_proj_domain_s20261022.json"
    if not tab.is_file():
        log("  补跑 p01_table_proj_domain.py（seed 20261022）...")
        run([PY, "-u", "scripts/p01_table_proj_domain.py", "--cnn-run-id", BASE_UNIFORM,
             "--label", "seed 20261022, 100-epoch", "--out-suffix", "_s20261022"],
            LOG_DIR / "table_s20261022.log")


def do_uniform(seed: int, summary: dict) -> None:
    if seed == 20261022:
        run_id = BASE_UNIFORM
        log(f"[uniform seed={seed}] 复用已发布 run={run_id}（不重跑）")
    else:
        run_id = newest_run_after_train(seed, domain="uniform")
    summary["uniform"][str(seed)] = uniform_metrics(run_id, seed)
    save(summary)


def do_flaw(seed: int, summary: dict) -> None:
    if seed == 20261022:
        run_id = BASE_FLAW
        log(f"[flaw seed={seed}] 复用已发布 run={run_id}（不重跑）")
    else:
        run_id = newest_run_after_train(seed, domain="flaw")
    summary["flaw"][str(seed)] = flaw_metrics(run_id)
    save(summary)


def newest_run_after_train(seed: int, domain: str) -> str:
    before = {p.name for p in RUNS.iterdir()
              if re.fullmatch(r"\d{8}_\d{4}_cnn_bs8_lr1e-4", p.name)}
    if domain == "uniform":
        cmd = [PY, "-u", "train.py", "--epochs", "100", "--seed", str(seed)]
    else:
        cmd = [PY, "-u", "train.py", "--csv", "sparse_targets_flaw.csv",
               "--config", "config_flaw.json", "--batch", "8",
               "--epochs", "100", "--seed", str(seed)]
    t0 = time.time()
    log(f"  训练 {domain} seed={seed} ...")
    run(cmd, LOG_DIR / f"train_{domain}_s{seed}.log")
    log(f"  训练完成 {time.time() - t0:.0f}s")
    after = {p.name for p in RUNS.iterdir()
             if re.fullmatch(r"\d{8}_\d{4}_cnn_bs8_lr1e-4", p.name)}
    new = sorted(after - before)
    if len(new) != 1:
        raise RuntimeError(f"预期新增 1 个 run，实得 {new}")
    rid = new[0]

    if domain == "uniform":
        for script, extra in (("scripts/cnn_recon.py", ["--cnn-run-id", rid]),
                              ("scripts/baseline_linear.py", ["--cnn-run-id", rid]),
                              ("scripts/baseline_linear_image.py", ["--cnn-run-id", rid])):
            t0 = time.time()
            log(f"  {Path(script).name} ...")
            run([PY, "-u", script] + extra,
                LOG_DIR / f"{Path(script).stem}_s{seed}.log")
            log(f"    {time.time() - t0:.0f}s")
        # 投影域 row16（线性那一份要现读正弦图，单独跑表格脚本）
        run([PY, "-u", "scripts/p01_table_proj_domain.py", "--cnn-run-id", rid,
             "--label", f"seed {seed}, 100-epoch", "--out-suffix", f"_s{seed}"],
            LOG_DIR / f"table_s{seed}.log")
    else:
        t0 = time.time()
        log("  p01_flaw_step3_eval ...")
        run([PY, "-u", "scripts/p01_flaw_step3_eval.py", "--run-id", rid,
             "--ref-run-id", BASE_UNIFORM], LOG_DIR / f"flaw_eval_s{seed}.log")
        log(f"    {time.time() - t0:.0f}s")
        # 匹配滤波（三源对比）是独立脚本，`p01_flaw_step3_eval.py` 不产出 mf_methods.json
        log("  p01_mf_methods ...")
        run([PY, "-u", "scripts/p01_mf_methods.py", "--flaw-run-id", rid],
            LOG_DIR / f"mf_s{seed}.log")
    return rid


def save(summary: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
                   encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[20261022, 20261023, 20261024])
    ap.add_argument("--domain", choices=["uniform", "flaw", "both"], default="both")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    summary = load(OUT) if OUT.is_file() else {"uniform": {}, "flaw": {}}
    summary.setdefault("uniform", {})
    summary.setdefault("flaw", {})

    if not args.aggregate_only:
        if args.domain in ("uniform", "both"):
            ensure_base_uniform()
        for seed in args.seeds:
            if args.domain in ("uniform", "both") and str(seed) not in summary["uniform"]:
                do_uniform(seed, summary)
            if args.domain in ("flaw", "both") and str(seed) not in summary["flaw"]:
                do_flaw(seed, summary)

    summary["stats"] = aggregate(summary)
    save(summary)
    log(f"汇总已写出: {OUT}")
    for domain, st in summary["stats"].items():
        print(f"\n===== {domain} =====")
        for k, v in st.items():
            print(f"  {k:22s} mean={v['mean']:.6f}  std={v['std']:.6f}  "
                  f"CI95=±{v['ci95_halfwidth']:.6f}  n={v['n']}  "
                  f"vals={[round(x, 6) for x in v['values']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
