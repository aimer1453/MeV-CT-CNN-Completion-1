#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/p02_sweep_eval.py —— 角度采样跨度扫描（sensitivity analysis）评估

任务
----
对给定采样跨度 s（2/4/6/8°），在同一口径下同时给出三种方法的
**投影域**（row16、all32）与**图像域**（RMSE / PSNR / SSIM）指标，
用来回答"CNN 是否随跨度增大而反超线性插值"。

几何（与 scripts/p02_make_step_csv.py 严格一致）
------------------------------------------------
半跨 h = s/2。目标 θ 的输入是 θ−3h、θ−h、θ+h，三者都落在"实测陪集"
(θ mod s) + h (mod s) 内。
  · 实测视角数 M = 360/s（陪集 c）
  · 待补视角数 M = 360/s（陪集 c+h）
  · 补全后正弦图 = 2M 个视角，角间距 s/2 度
        s=2 → 360 视角 / 1.0°    s=4 → 180 视角 / 2.0°
        s=6 → 120 视角 / 3.0°    s=8 →  90 视角 / 4.0°
  · 方向数 = s（每个陪集轮流当一次"实测"）。s=2 的两个方向就是原 phase A/B。

同一 s 下三种方法用**完全相同**的视角集合，因此可以直接横比：
  fbp_sparse  只用 M 个实测视角做 FBP（不补全）
  linear      ½·[L(θ−h) + L(θ+h)] 补全 → 2M 视角 FBP
  cnn         best.pt 预测补全 → 2M 视角 FBP
  dense 参考  360 个视角全部实测的 hamming FBP（与 baseline_recon.py 同式）

投影域口径
----------
  只在 **test split** 的目标视角上统计（每个 θ 恰好在一个方向里当一次目标，
  故"遍历全部方向"恰好把每个 test θ 覆盖一次，不重不漏）。
  row16 = 只算 RECON_ROW=16（真正进入正弦图的那一行）
  all32 = 全 32 行 mask 加权（与 train.py evaluate 同式）
  mask  = (I_target > 0) & (empty_proj > 0)，逐像素
  池化  = sqrt( Σ(误差²) / Σ(mask 像素数) )，先逐能量再跨能量

图像域口径
----------
  dense 参考、FOV 圆（r<=600）、SSIM 中心 848×848、`psnr_common`（四档统一
  data_range）全部复用 scripts/baseline_recon.py 的实现/常量。
  逐方向 × 逐能量各一次指标，再按方向、能量取均值（每个方向像素数相同）。

用法
----
  python scripts/p02_sweep_eval.py --step 2 --cnn-run 20261023_2332_cnn_bs8_lr1e-4
  python scripts/p02_sweep_eval.py --step 8 --cnn-run <run_id> --limit-energy 1.5
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

import baseline_recon as br                        # noqa: E402
from data.dataset import log_ratio, valid_mask     # noqa: E402

OUT_DIR = ROOT / "runs" / "_reviewer_p0"
FILTER = "hamming"
BATCH = 16
ENERGIES = ("1.5", "3", "4.5", "6")


# ------------------------------------------------------------------ 数据
def load_angles() -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {}
    with open(resolve_csv("angles.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["energy_MeV"] in ENERGIES:
                out.setdefault(r["energy_MeV"], {})[int(round(float(r["angle_deg"])))] = r["file"]
    return out


def load_split() -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    with open(resolve_csv("sparse_targets.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["energy_MeV"], int(round(float(r["target_angle_deg"]))))] = r["split"]
    return out


class EnergyStack:
    """一个能量档：32 行 -log 栈 + 原始投影（mask 用） + empty。"""

    def __init__(self, energy: str, files: dict[int, str], fill: float,
                 empty_path: Path | None = None) -> None:
        self.energy = energy
        if empty_path is None:
            # 默认路径（主数据域）：angles.csv 的 file 形如 3_3/<scan>/projections/x.npy
            scan = Path(files[0]).parent.parent.name
            empty_path = ROOT / "3_3" / scan / "results" / "empty_proj.npy"
        # 缺陷批次（3_3_flaw_1_2/ 等）不在 3_3/ 下，由调用方显式传入
        self.empty = np.load(empty_path).astype(np.float32)
        n = len(files)
        self.proj = np.empty((n, 32, 1200), dtype=np.float32)
        for a in range(n):
            self.proj[a] = np.load(ROOT / files[a]).astype(np.float32)
        L = np.empty((n, 32, 1200), dtype=np.float32)
        for a in range(n):
            l = log_ratio(self.proj[a], self.empty)
            l[~np.isfinite(l)] = fill
            L[a] = l
        self.L = L                                   # (360,32,1200) 已填充
        self.row16 = L[:, br.RECON_ROW, :].astype(np.float64)
        self.theta = np.arange(n, dtype=np.float64)
        self.valid = np.empty((n, 32, 1200), dtype=bool)
        for a in range(n):
            self.valid[a] = valid_mask(self.proj[a], self.empty)


# ------------------------------------------------------------------ CNN
def channel(img: np.ndarray, empty: np.ndarray, fill: float,
            mean: float, std: float) -> np.ndarray:
    """单张投影 -> 归一化 -log 图（与 data/dataset.py::_channel 逐字同式）。"""
    L = log_ratio(img, empty).astype(np.float64)
    L[~np.isfinite(L)] = fill
    return ((L - mean) / std).astype(np.float32)


def predict(model, device, items, stacks: dict[str, "EnergyStack"],
            files: dict[str, dict[int, str]], step: int,
            mean: float, std: float, fill: float, n_views: int = 3) -> np.ndarray:
    """items: [(energy, theta)]，返回 (N,32,1200) 已反归一化的物理域 -log。

    `n_views` 只影响输入装配，**不改变 n_views=3 的任何既有行为**（默认即 3）：

      · n_views=3（正文）：[θ−1°, θ−3°, θ+1°] + sin/cos → 5 通道
      · n_views=2（消融，与 `data/dataset.py` 逐字同序）：丢掉 θ−3°，
        [θ−1°, θ+1°] + sin/cos → 4 通道
    """
    import torch

    if n_views not in (2, 3):
        raise ValueError(f"n_views 仅允许 2 或 3，收到 {n_views!r}")
    h = step // 2
    offs = (-h, -3 * h, +h) if n_views == 3 else (-h, +h)
    out = np.empty((len(items), 32, 1200), dtype=np.float32)
    for i0 in range(0, len(items), BATCH):
        chunk = items[i0:i0 + BATCH]
        xs = np.empty((len(chunk), n_views + 2, 32, 1200), dtype=np.float32)
        for j, (e, th) in enumerate(chunk):
            empty = stacks[e].empty
            for k, off in enumerate(offs):
                a = (th + off) % 360
                img = np.load(ROOT / files[e][a]).astype(np.float32)
                xs[j, k] = channel(img, empty, fill, mean, std)
            t = math.radians(float(th))
            xs[j, n_views] = math.sin(t)
            xs[j, n_views + 1] = math.cos(t)
        with torch.no_grad():
            pred = model(torch.from_numpy(xs).to(device)).cpu().numpy()
        out[i0:i0 + len(chunk)] = (pred[:, 0] * std + mean).astype(np.float32)
    return out


# ------------------------------------------------------------------ 主流程
def run(args: argparse.Namespace) -> dict:
    step = args.step
    h = step // 2
    cfg = json.loads((br.RUNS_DIR / args.cnn_run / "config.json").read_text(encoding="utf-8"))
    norm = cfg["norm"]
    mean, std, fill = float(norm["norm_mean"]), float(norm["norm_std"]), float(norm["zero_log_fill"])

    files = load_angles()
    split = load_split()
    energies = [e for e in ENERGIES if not args.limit_energy or e == args.limit_energy]

    stacks: dict[str, EnergyStack] = {}
    for e in energies:
        t0 = time.time()
        stacks[e] = EnergyStack(e, files[e], fill)
        print(f"[{e} MeV] 载入 360×32×1200 完成 {time.time() - t0:.1f}s", flush=True)

    # dense 参考
    dense: dict[str, np.ndarray] = {}
    for e in energies:
        st = stacks[e]
        dense[e] = br.restrict_to_fov(br.recon_fbp(st.row16, st.theta, FILTER))
    dr_common = max(float(dense[e].max() - dense[e].min()) for e in energies)

    # CNN
    import torch
    from models.unet import UNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    ck = torch.load(br.RUNS_DIR / args.cnn_run / "best.pt", map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[CNN] {args.cnn_run}  epoch={ck['epoch']}  "
          f"val_rmse_valid_phys={ck['val_rmse_valid_phys']:.6f}", flush=True)

    mask_fov = br.fov_mask()
    per_sample: list[dict] = []
    # 投影域累加器：method -> roi -> energy -> (se, n)
    acc: dict[str, dict[str, dict[str, list]]] = {
        m: {r: {e: [0.0, 0] for e in energies} for r in ("row16", "all32")}
        for m in ("cnn", "linear", "cubic")}

    for c in range(step):
        for e in energies:
            st = stacks[e]
            meas = [a for a in range(360) if a % step == c]
            targ = [a for a in range(360) if a % step == (c + h) % step]
            assert len(meas) == len(targ) == 360 // step

            # ---- 线性补全 & 三次插值补全 & CNN 补全（全部 32 行）
            lin32 = np.empty((len(targ), 32, 1200), dtype=np.float64)
            cub32 = np.empty((len(targ), 32, 1200), dtype=np.float64)
            for i, th in enumerate(targ):
                # 线性：只用夹住 θ 的两帧
                lin32[i] = 0.5 * (st.L[(th - h) % 360] + st.L[(th + h) % 360])
                # 三次（Lagrange 四点等距，节点 θ±0.5s、θ±1.5s，在中点 θ 求值）
                #   L(θ) = 9/16·[L(θ-h) + L(θ+h)] - 1/16·[L(θ-3h) + L(θ+3h)]
                #   四个节点都落在"实测陪集"内（≡ c+h mod s），故都是真测量值。
                #   注意：cubic 用了 4 个实测视角，比 CNN 的输入（3 个）多一个，
                #        是**被优待**的基线；CNN 若仍能赢，结论更强。
                cub32[i] = (9.0 / 16.0) * (st.L[(th - h) % 360] + st.L[(th + h) % 360]) \
                    - (1.0 / 16.0) * (st.L[(th - 3 * h) % 360] + st.L[(th + 3 * h) % 360])
            cnn32 = predict(model, device, [(e, th) for th in targ], stacks, files,
                            step, mean, std, fill).astype(np.float64)

            # ---- 三种正弦图
            def compose(fill_rows: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
                """None = 不补全（只实测）。否则按 targ 顺序填入。"""
                if fill_rows is None:
                    ang = np.array(meas, dtype=float)
                    return st.row16[meas], ang
                sino = st.row16.copy()
                ang = np.arange(360, dtype=float)
                for i, th in enumerate(targ):
                    sino[th] = fill_rows[i, br.RECON_ROW, :]
                sel = sorted(meas + targ)
                return sino[sel], ang[sel]

            images = {}
            s_sparse, th_sparse = compose(None)
            s_lin, th_lin = compose(lin32)
            s_cub, th_cub = compose(cub32)
            s_cnn, th_cnn = compose(cnn32)
            for name, (s_, th_) in (("fbp_sparse", (s_sparse, th_sparse)),
                                    ("linear", (s_lin, th_lin)),
                                    ("cubic", (s_cub, th_cub)),
                                    ("cnn", (s_cnn, th_cnn))):
                img = br.restrict_to_fov(br.recon_fbp(s_, th_, FILTER))
                m = br.image_metrics(dense[e], img, mask_fov)
                per_sample.append({
                    "energy_MeV": e, "direction": c, "method": name,
                    "rmse_valid": m["rmse_valid"], "rmse_full": m["rmse_full"],
                    "psnr": m["psnr"], "ssim": m["ssim"],
                    "psnr_common": 10.0 * math.log10(dr_common * dr_common
                                                     / (m["rmse_valid"] ** 2)),
                    "n_views": int(s_.shape[0]),
                })
            # ---- 投影域：只统计 test split 的目标
            for i, th in enumerate(targ):
                if split[(e, th)] != "test":
                    continue
                m16 = st.valid[th, br.RECON_ROW, :]
                m32 = st.valid[th]
                for meth, pred in (("cnn", cnn32[i]), ("linear", lin32[i]),
                                   ("cubic", cub32[i])):
                    for roi, (gt, pr, mk) in (
                            ("row16", (st.row16[th], pred[br.RECON_ROW], m16)),
                            ("all32", (st.L[th].astype(np.float64), pred, m32))):
                        if mk.any():
                            acc[meth][roi][e][0] += float(((pr - gt) ** 2)[mk].sum())
                            acc[meth][roi][e][1] += int(mk.sum())
        print(f"  [s={step}] 方向 {c + 1}/{step} 完成", flush=True)

    # ---- 汇总
    def agg(method: str) -> dict:
        sub = [s for s in per_sample if s["method"] == method]
        per_e = {}
        for e in energies:
            se = [s for s in sub if s["energy_MeV"] == e]
            mse = float(np.mean([s["rmse_valid"] ** 2 for s in se]))
            per_e[e] = {
                "n_directions": len(se),
                "n_views_completed": se[0]["n_views"],
                "rmse_valid": math.sqrt(mse),
                "rmse_full": float(np.mean([s["rmse_full"] for s in se])),
                "psnr": float(np.mean([s["psnr"] for s in se])),
                "psnr_common": float(np.mean([s["psnr_common"] for s in se])),
                "ssim": float(np.mean([s["ssim"] for s in se])),
            }
        mse_all = float(np.mean([s["rmse_valid"] ** 2 for s in sub]))
        return {
            "per_energy": per_e,
            "overall": {
                "n_samples": len(sub),
                "n_views_completed": sub[0]["n_views"],
                "rmse_valid": math.sqrt(mse_all),
                "rmse_full": float(np.mean([s["rmse_full"] for s in sub])),
                "psnr": float(np.mean([s["psnr"] for s in sub])),
                "psnr_common": float(np.mean([s["psnr_common"] for s in sub])),
                "ssim": float(np.mean([s["ssim"] for s in sub])),
            },
        }

    proj = {}
    for meth in ("cnn", "linear", "cubic"):
        proj[meth] = {}
        for roi in ("row16", "all32"):
            per_e = {}
            se = n = 0.0
            for e in energies:
                s_, n_ = acc[meth][roi][e]
                per_e[e] = {"rmse_valid": math.sqrt(s_ / n_) if n_ else float("nan"),
                            "n_valid_pixels": int(n_)}
                se += s_
                n += n_
            proj[meth][roi] = {"per_energy": per_e,
                               "overall": {"rmse_valid": math.sqrt(se / n) if n else float("nan"),
                                           "n_valid_pixels": int(n)}}

    return {
        "step_deg": step,
        "half_step_deg": h,
        "n_measured_views": 360 // step,
        "n_completed_views": 720 // step,
        "completed_view_spacing_deg": step / 2.0,
        "n_directions": step,
        "cnn_run_id": args.cnn_run,
        "cnn_best_epoch": int(ck["epoch"]),
        "filter": FILTER,
        "dense_reference": "360 视角全实测 hamming FBP（与 baseline_recon.py 同式）",
        "data_range_common": dr_common,
        "image": {m: agg(m) for m in ("fbp_sparse", "linear", "cubic", "cnn")},
        "projection": proj,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="角度采样跨度扫描评估")
    ap.add_argument("--step", type=int, required=True, choices=(2, 4, 6, 8, 10, 12))
    ap.add_argument("--cnn-run", required=True, help="runs/<id>，该跨度下训练出的 run")
    ap.add_argument("--limit-energy", default=None, help="只跑一档能量（快速试跑）")
    ap.add_argument("--tag", default="", help="输出文件名中段，如 60 → sweep60_s8.json")
    ap.add_argument("--out", default=None, help="输出 json 路径")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = run(args)
    path = Path(args.out) if args.out else OUT_DIR / f"sweep{args.tag}_s{args.step}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"s={args.step}°  实测 {res['n_measured_views']} 视角  "
          f"补全后 {res['n_completed_views']} 视角（间距 {res['completed_view_spacing_deg']}°）")
    for roi in ("row16", "all32"):
        c = res["projection"]["cnn"][roi]["overall"]["rmse_valid"]
        l = res["projection"]["linear"][roi]["overall"]["rmse_valid"]
        q = res["projection"]["cubic"][roi]["overall"]["rmse_valid"]
        print(f"  投影域 {roi:6s}: CNN {c:.6f} | linear {l:.6f} | cubic {q:.6f} "
              f"| CNN/linear = {c / l:.4f} | CNN/cubic = {c / q:.4f}")
    for m in ("fbp_sparse", "linear", "cubic", "cnn"):
        o = res["image"][m]["overall"]
        print(f"  图像域 {m:11s}: rmse {o['rmse_valid']:.6f} | psnr_common {o['psnr_common']:.4f}"
              f" | ssim {o['ssim']:.6f}")
    print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
