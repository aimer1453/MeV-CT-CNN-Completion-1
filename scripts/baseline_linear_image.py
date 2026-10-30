#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/baseline_linear_image.py —— 线性插值基线的**图像域**版本（Table 4 第四行）

为什么需要
----------
论文侧 §5.6 的 Table 4 有四行：`Linear interpolation / FBP / SART / CNN completion`。
R7 把 Table 4 改成**一张图像域表**（三方法 + dense 参考），但 **linear_interp 原本只有
投影域数字**（`runs/<id>_linear_interp/`，domain = "projection (-log physical)"），
直接搬进图像域表是混域、不可比。本脚本补上它的图像域版本。

方法定义（与投影域基线**逐字一致**，见 `scripts/baseline_linear.py`）
--------------------------------------------------------------------
    target 投影 = (P[ch0] + P[ch_next]) / 2，两点算术平均；
    target 恰为 ch0 与 ch_next 的中点，故线性插值退化为均值。

其中 `ch0 = (target-1) % 360`、`ch_next = (target+1) % 360`，两者都落在**实测相位**内
（target 在相位 B 时，θ−1 与 θ+1 都是偶数角 = 相位 A = 实测），所以这个替换是
**合法的部署行为**（只用实测数据，没有任何 GT 泄漏）。

流程（与 `scripts/cnn_recon.py` 完全对称，只把"预测"换成"两点均值"）
--------------------------------------------------------------------
  1. 取某相位 P 的 180 个视角作为实测输入；
  2. 把 P 的正弦图按 `(P[θ−1] + P[θ+1]) / 2` 填出 opposite(P) 的 180 个视角；
  3. 交错成 360 视角稠密正弦图 → `iradon(hamming, circle=True)` → 1200×1200；
  4. 与 dense（360 视角全实测 FBP）比 PSNR / SSIM / RMSE，**同一份代码、同一滤镜、
     同一裁剪、同一 dense 参考**（直接 import `baseline_recon`）。

口径
----
  - `domain: "image"`；恒有 `rmse_full <= rmse_valid`（`circle=True` 把圆外两图都置零），
    与投影域相反。
  - `coverage` / `valid_coverage` 在图像域不适用，填 `NaN`。
  - PSNR 逐档 `psnr` 用各档自己的 data_range；跨能量比较必须用 `psnr_common`
    （四档统一 data_range），与 `baseline_recon.py` 同口径。

输出
----
  runs/<cnn_run_id>_linear_interp_image/{config.json,train_log.csv,
                                        test_metrics.json,test_per_sample.csv}
  runs/<cnn_run_id>_linear_interp_image/recon/linear_<e>_<phase>.npy（.npy 被 ignore）

用法
----
  conda run -n ct_full python scripts/baseline_linear_image.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import baseline_recon as br                       # noqa: E402

SPARSE_TARGETS = resolve_csv("sparse_targets.csv")
RUNS_DIR = ROOT / "runs"
DEFAULT_CNN_RUN_ID = "20261022_2352_cnn_bs8_lr1e-4"
NAN = float("nan")


def load_energy_rows(energy: str) -> list[dict]:
    with open(SPARSE_TARGETS, newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        return [x for x in r if x["energy_MeV"] == energy]


def run(energy: str, fill: float, filter_name: str, recon_dir: Path) -> dict:
    rows = load_energy_rows(energy)
    by_phase = {p: [x for x in rows if x["phase"] == p] for p in ("A", "B")}
    for p in ("A", "B"):
        if len(by_phase[p]) != 180:
            raise br.ReconSpecError(f"[{energy} MeV] 相位 {p} 期望 180 行，实得 {len(by_phase[p])}")

    sino, theta = br.build_sinogram(energy, ROOT / "3_3", fill)   # (360,1200) 实测 -log
    if not np.array_equal(np.round(theta).astype(np.int64), np.arange(360)):
        raise br.ReconSpecError(f"[{energy} MeV] theta 不是 0..359 的整数序列")
    idx = {int(a): i for i, a in enumerate(np.round(theta).astype(np.int64))}

    dense = br.get_dense_reference(energy, ROOT / "3_3", fill, filter_name,
                                   recon_dir, reuse=True)

    out: dict = {"energy": energy, "dense": dense, "recon": {}}
    for in_phase in ("A", "B"):
        missing = br.opposite_phase(in_phase)
        t0 = time.time()
        sino_comp = sino.copy()
        for r in by_phase[missing]:
            a = int(float(r["target_angle_deg"]))
            lo = sino[idx[(a - 1) % 360]]        # ch0  = θ−1°，落在实测相位
            hi = sino[idx[(a + 1) % 360]]        # ch_next = θ+1°，落在实测相位
            sino_comp[idx[a]] = 0.5 * (lo + hi)  # 线性插值退化为两点均值
        img = br.restrict_to_fov(br.recon_fbp(sino_comp, theta, filter_name))
        np.save(recon_dir / f"linear_{energy}_{in_phase}.npy", img.astype(np.float32))
        out["recon"][in_phase] = img
        print(f"  [{energy} MeV] 输入相位 {in_phase} → 插值填充相位 {missing}："
              f"180 视角，{time.time() - t0:.1f}s", flush=True)
    return out


def write_outputs(results: list[dict], cnn_run_id: str, fill: float,
                  filter_name: str) -> Path:
    out_dir = RUNS_DIR / f"{cnn_run_id}_linear_interp_image"
    mask = br.fov_mask()
    commit, dirty = br.git_info()

    test_rows = br.load_test_rows()
    per_sample: list[dict] = []
    for r in test_rows:
        e = r["energy_MeV"]
        in_phase = br.opposite_phase(r["phase"])
        res = next(x for x in results if x["energy"] == e)
        m = br.image_metrics(res["dense"], res["recon"][in_phase], mask)
        per_sample.append({
            "energy_MeV": e,
            "target_angle_deg": float(r["target_angle_deg"]),
            "split": "test",
            "n_valid_pixels": m["n_valid_pixels"],
            "coverage": NAN, "valid_coverage": NAN,
            "rmse_valid": m["rmse_valid"], "rmse_full": m["rmse_full"],
            "psnr": m["psnr"], "ssim": m["ssim"],
            "rmse_invariant_warn": False,
            "target_phase": r["phase"], "input_phase": in_phase,
        })

    dr_common = max(float(x["dense"].max() - x["dense"].min()) for x in results)
    per_energy: dict[str, dict] = {}
    for e in br.ENERGIES:
        sub = [s for s in per_sample if s["energy_MeV"] == e]
        nv = sum(s["n_valid_pixels"] for s in sub)
        se_v = sum(s["rmse_valid"] ** 2 * s["n_valid_pixels"] for s in sub)
        mse_v = se_v / nv
        res = next(x for x in results if x["energy"] == e)
        per_energy[e] = {
            "n_samples": len(sub),
            "coverage": NAN, "valid_coverage": NAN,
            "rmse_valid": math.sqrt(mse_v),
            "rmse_full": math.sqrt(sum(s["rmse_full"] ** 2 for s in sub) / len(sub)),
            "psnr": float(np.mean([s["psnr"] for s in sub])),
            "psnr_common": 10.0 * math.log10(dr_common * dr_common / mse_v),
            "ssim": float(np.mean([s["ssim"] for s in sub])),
            "data_range": float(res["dense"].max() - res["dense"].min()),
            "data_range_common": dr_common,
        }
    n_all = sum(v["n_samples"] for v in per_energy.values())
    n_v = sum(s["n_valid_pixels"] for s in per_sample)
    se_v = sum(s["rmse_valid"] ** 2 * s["n_valid_pixels"] for s in per_sample)
    overall = {
        "n_samples": n_all, "coverage": NAN, "valid_coverage": NAN,
        "rmse_valid": math.sqrt(se_v / n_v),
        "rmse_full": math.sqrt(sum(v["rmse_full"] ** 2 * v["n_samples"]
                                   for v in per_energy.values()) / n_all),
        "psnr": float(np.mean([v["psnr"] for v in per_energy.values()])),
        "psnr_common": float(np.mean([v["psnr_common"] for v in per_energy.values()])),
        "ssim": float(np.mean([v["ssim"] for v in per_energy.values()])),
    }
    if overall["rmse_full"] > overall["rmse_valid"] + 1e-9:
        raise br.ReconSpecError("图像域总体 rmse_full > rmse_valid，圆外零误差假设被破坏")

    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": out_dir.name,
        "method": "linear_interp_completed",
        "domain": "image",
        "method_note": "opposite 相位的 180 个视角用 (P[θ−1°] + P[θ+1°]) / 2 填充"
                       "（与投影域基线 scripts/baseline_linear.py 逐字同一公式；"
                       "θ−1°、θ+1° 落在实测相位内，无 GT 泄漏）"
                       "→ 与 180 个实测视角交错成 360 视角正弦图 → hamming FBP",
        "reference_run_id": cnn_run_id,
        "git_commit": commit, "git_dirty": dirty,
        "geometry": {"sensor_x": br.OUTPUT_SIZE, "sensor_y": 32,
                     "recon_row": br.RECON_ROW, "output_size": [br.OUTPUT_SIZE] * 2,
                     "sinogram_shape": [360, br.OUTPUT_SIZE]},
        "n_views": 360, "n_completed": 180, "held_out_restricted": False,
        "recon": {"filter_name": filter_name, "circle": True,
                  "fov_restriction": "圆外置零（circle=True 自动）"},
        "metric_definition": {
            "domain": "image",
            "rmse_valid": f"FOV 重建圆（r<={br.CIRCLE_RADIUS}，{int(mask.sum())} 像素）上的 RMSE",
            "rmse_full": "全部 1200x1200；圆外两图皆 0，故恒有 rmse_full <= rmse_valid",
            "psnr": "10*log10(data_range^2/mse_valid)，data_range 逐档不同",
            "psnr_common": "同上但 data_range 取四档统一值，跨能量比较必须用这个",
            "ssim": f"中心 {br.SSIM_CROP}x{br.SSIM_CROP} 裁剪上的 SSIM",
            "reference": "dense = 360 视角全实测 FBP，与 baseline_recon.py 同一份代码",
            "cross_domain_warning": "本文件 rmse_valid 是图像域，与投影域 rmse_valid 不同域、不可同表",
        },
        "data": {"targets": "sparse_targets.csv", "split": "test",
                 "n_samples": n_all, "energies_MeV": list(br.ENERGIES)},
        "note": "本条为 **R7 规格之外的新增行**：R7 §7.4 只列了三方法 + dense，"
                "但论文 Table 4 原本有 Linear interpolation 行（且只有投影域数字），"
                "整表改图像域后该行会缺失，故补此版本。是否进 Table 4 由论文侧决定。",
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with open(out_dir / "train_log.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=br.LOG_COLUMNS)
        w.writeheader()
        for e in br.ENERGIES:
            v = per_energy[e]
            w.writerow({"epoch": -1, "split": "test", "energy_MeV": e,
                        "n_samples": v["n_samples"], "coverage": "NaN",
                        "valid_coverage": "NaN",
                        "rmse_valid": f"{v['rmse_valid']:.6f}",
                        "rmse_full": f"{v['rmse_full']:.6f}",
                        "psnr": f"{v['psnr']:.6f}", "ssim": f"{v['ssim']:.6f}",
                        "n_invariant_warn": 0, "lr": "NaN",
                        "loss_train": "NaN", "loss_val": "NaN"})

    with open(out_dir / "test_per_sample.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=br.PER_SAMPLE_COLUMNS)
        w.writeheader()
        for r in per_sample:
            row = {k: r[k] for k in br.PER_SAMPLE_COLUMNS}
            for k in ("coverage", "valid_coverage"):
                row[k] = "NaN"
            for k in ("rmse_valid", "rmse_full", "psnr", "ssim"):
                row[k] = f"{row[k]:.6f}"
            row["target_angle_deg"] = f"{float(row['target_angle_deg']):g}"
            w.writerow(row)

    tm = {
        "run_id": out_dir.name,
        "method": "linear_interp_completed",
        "domain": "image",
        "n_views": 360, "n_completed": 180, "held_out_restricted": False,
        "git_commit": commit, "git_dirty": dirty,
        "reference_run_id": cnn_run_id,
        "reference_definition": "dense = 360 视角全实测 hamming FBP，与 baseline_recon.py 同代码同参数",
        "invariant_warn": {"n_warn": 0, "n_samples": n_all, "ratio": 0.0,
                           "note": "图像域恒有 rmse_full <= rmse_valid（圆外两图皆 0），"
                                   "与投影域相反；校验在写入前 raise"},
        "overall": overall,
        "per_energy": per_energy,
        "per_sample": {f"{r['energy_MeV']}@{r['target_angle_deg']:g}": {
            "target_phase": r["target_phase"], "input_phase": r["input_phase"],
            "rmse_valid": r["rmse_valid"], "rmse_full": r["rmse_full"],
            "psnr": r["psnr"], "ssim": r["ssim"]} for r in per_sample},
        "cross_domain_warning": "domain=image：rmse_valid 为图像域，禁止与投影域 rmse_valid 同表；"
                                "coverage/valid_coverage 在图像域不适用，填 NaN",
    }
    (out_dir / "test_metrics.json").write_text(
        json.dumps(tm, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="线性插值基线的图像域版本（Table 4 第四行）")
    ap.add_argument("--cnn-run-id", default=DEFAULT_CNN_RUN_ID)
    ap.add_argument("--filter-name", default="hamming")
    args = ap.parse_args(argv)

    try:
        cfg_p = RUNS_DIR / args.cnn_run_id / "config.json"
        if not cfg_p.is_file():
            raise br.ReconSpecError(f"缺少 {cfg_p}")
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        norm = cfg.get("norm") or {}
        for k in ("norm_mean", "norm_std", "zero_log_fill"):
            if norm.get(k) is None:
                raise br.ReconSpecError(f"config.json 的 norm.{k} 缺失，禁止用默认值")
        fill = float(norm["zero_log_fill"])

        out_dir = RUNS_DIR / f"{args.cnn_run_id}_linear_interp_image"
        recon_dir = out_dir / "recon"
        recon_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for e in br.ENERGIES:
            print(f"[{e} MeV] 开始", flush=True)
            results.append(run(e, fill, args.filter_name, recon_dir))

        # dense 参考必须与 fbp_sparse 的缓存逐字节一致
        ref_dir = RUNS_DIR / f"{args.cnn_run_id}_fbp_sparse" / "recon"
        for x in results:
            p = ref_dir / f"dense_{x['energy']}_fbp_{args.filter_name}.npy"
            if p.is_file():
                same = bool(np.array_equal(np.load(p).astype(np.float32),
                                           x["dense"].astype(np.float32)))
                if not same:
                    raise br.ReconSpecError(f"[{x['energy']} MeV] dense 参考与 fbp_sparse 缓存不一致")
        print("dense 参考与 fbp_sparse 缓存逐字节一致 ✓", flush=True)

        out_dir = write_outputs(results, args.cnn_run_id, fill, args.filter_name)
    except br.ReconSpecError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:                             # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"[FAIL] 异常: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    tm = json.loads((out_dir / "test_metrics.json").read_text(encoding="utf-8"))
    print("=" * 96)
    print("method = linear_interp_completed   domain = image")
    print("-" * 96)
    print(f"{'energy':>8} {'n':>4} {'rmse_valid':>12} {'rmse_full':>12} "
          f"{'psnr':>9} {'psnr_common':>12} {'ssim':>9}")
    for e in br.ENERGIES:
        v = tm["per_energy"][e]
        print(f"{e:>8} {v['n_samples']:>4} {v['rmse_valid']:>12.6f} "
              f"{v['rmse_full']:>12.6f} {v['psnr']:>9.4f} {v['psnr_common']:>12.4f} "
              f"{v['ssim']:>9.6f}")
    o = tm["overall"]
    print(f"{'overall':>8} {o['n_samples']:>4} {o['rmse_valid']:>12.6f} "
          f"{o['rmse_full']:>12.6f} {o['psnr']:>9.4f} {o['psnr_common']:>12.4f} "
          f"{o['ssim']:>9.6f}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
