#!/usr/bin/env python
"""P0-1 评测：同一份数据表上比较 CNN 补全 / 线性插值 / 三次插值。

协议（与 cnn_recon.py 逐项对齐）
-------------------------------
每个能量档做两次"半扫描"实验：
    输入相位 P（180 个实测视角）→ 预测 opposite(P) 的 180 个视角 → 拼 360 视角 → FBP
两个方向各算一套指标，报两者平均（逐相位值同时保留在 JSON 里）。

三套口径（全部在 -log 物理域）
-----------------------------
  row16  : 只算探测器第 16 行（RECON_ROW，真正进入重建的那一行）
  all32  : 全 32 行 per-pixel mask（与 train.py::evaluate 的 rmse_valid 完全一致）
  image  : 用 row16 拼 360 视角 → iradon(hamming) → 与 dense 参考比 RMSE
  **三套指标都只在"被预测的那 180 个视角"上统计**，不混入实测视角。

dense 参考是**同一剂量档**的 360 视角全实测 FBP，故 image 指标衡量的是
"补全相对本档可达最优"的差距，而非"低剂量相对高剂量"的绝对退化。

插值基线的角度公式（不做任何 CSV 角度运算，直接用相邻实测角）
-----------------------------------------------------------
  实测相位 = in_phase（θ±1 与 θ±3 都落在同一相位）
  线性：P(θ) = ½[P(θ−1) + P(θ+1)]
  三次：4 点中点系数（x = −3,−1,+1,+3 等距）
        P(θ) = [−P(θ−3) + 9P(θ−1) + 9P(θ+1) − P(θ+3)] / 16

用法
----
    python scripts/dose_eval.py --factor 1.0 --csv sparse_targets.csv \
        --config config.json \
        --run-ids 20261024_0400_cnn_bs8_lr1e-4 20261024_1633_cnn_bs8_lr1e-4 \
                  20261024_1805_cnn_bs8_lr1e-4
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_recon as br                                    # noqa: E402
from data.dataset import (SparseCTDataset, load_csv_rows, log_ratio,  # noqa: E402
                          valid_mask)

RUNS = ROOT / "runs"
NAN = float("nan")


def load_cfg(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def norm_of(cfg: dict) -> tuple[float, float, float]:
    n = cfg["norm"]
    return float(n["norm_mean"]), float(n["norm_std"]), float(n["zero_log_fill"])


def read_rows(csv_path: Path) -> list[dict]:
    return list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))


def scan_root(rows: list[dict]) -> Path:
    return (ROOT / rows[0]["target_file"]).parent.parent.parent


def build_stack(rows: list[dict], root: Path, fill: float):
    """(360,32,1200) -log stack + theta + 有效 mask。"""
    rows = sorted(rows, key=lambda r: float(r["target_angle_deg"]))
    ang = np.array([float(r["target_angle_deg"]) for r in rows])
    if not np.array_equal(ang.astype(np.int64), np.arange(360)):
        raise SystemExit("[FATAL] 角度不是 0..359 的整数序列")
    sds = {r["scan_dir"] for r in rows}
    if len(sds) != 1:
        raise SystemExit(f"[FATAL] 多个 scan_dir: {sds}")
    e = np.load(root / sds.pop() / "results" / "empty_proj.npy").astype(np.float32)
    L = np.empty((360, 32, 1200))
    M = np.empty((360, 32, 1200), dtype=bool)
    for i, r in enumerate(rows):
        I = np.load(ROOT / r["target_file"]).astype(np.float32)
        l = log_ratio(I, e).astype(np.float64)
        m = valid_mask(I, e)
        l[~m] = fill
        L[i] = l
        M[i] = m
    return L, ang, M


def interp_linear(L: np.ndarray) -> np.ndarray:
    idx = np.arange(360)
    return 0.5 * (L[(idx - 1) % 360] + L[(idx + 1) % 360])


def interp_cubic(L: np.ndarray) -> np.ndarray:
    idx = np.arange(360)
    return (-L[(idx - 3) % 360] + 9.0 * L[(idx - 1) % 360]
            + 9.0 * L[(idx + 1) % 360] - L[(idx + 3) % 360]) / 16.0


def predict_phase(run_id: str, cfg_path: Path, csv_path: Path, energy: float,
                  phase: str, batch: int, device, single_row: bool = False) -> tuple[np.ndarray, list[dict]]:
    import torch
    from models.unet import UNet

    mean, std, fill = norm_of(load_cfg(cfg_path))
    ds = SparseCTDataset(split="train", zero_log_fill=fill, norm_mean=mean,
                         norm_std=std, n_views=3, csv_path=csv_path, data_root=ROOT,
                         single_row=single_row)
    all_rows = []
    for sp in ("train", "val", "test"):          # 补全三套划分：要预测的是全部 180 个视角
        all_rows += load_csv_rows(csv_path, sp)
    keep = [i for i, r in enumerate(all_rows)
            if abs(float(r["energy_MeV"]) - energy) < 1e-9 and r["phase"] == phase]
    if len(keep) != 180:
        raise SystemExit(f"[FATAL] {energy} MeV phase {phase} 目标 {len(keep)}，期望 180")
    ds.rows = [all_rows[i] for i in keep]

    ckpt = RUNS / run_id / "best.pt"
    if not ckpt.is_file():
        raise SystemExit(f"[FATAL] 缺 {ckpt}")
    model = UNet(in_channels=5, out_channels=1)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()

    out = np.empty((len(ds.rows), 32, 1200))
    with torch.no_grad():
        for i in range(0, len(ds.rows), batch):
            items = [ds[j] for j in range(i, min(i + batch, len(ds.rows)))]
            x = torch.from_numpy(np.stack([it["x"] for it in items])).to(device)
            out[i:i + len(items)] = model(x).cpu().numpy()[:, 0] * std + mean
    # 必须按**角度升序**返回：CSV 行序是 train/val/test 拼接，与 range(360) 不对齐
    ang = np.array([float(r["target_angle_deg"]) for r in ds.rows])
    order = np.argsort(ang)
    return out[order], [ds.rows[i] for i in order]


def rmse_pair(pred: np.ndarray, tgt: np.ndarray, mask: np.ndarray, row: int | None):
    """pred/tgt/mask 都是"只含被预测视角"的子集。返回 (rmse_valid, rmse_full)。"""
    d = pred - tgt
    if row is not None:
        d, mask = d[:, row, :], mask[:, row, :]
    se = float(((d ** 2) * mask).sum())
    n = int(mask.sum())
    sef = float((d ** 2).sum())
    nf = int(d.size)
    return (math.sqrt(se / n) if n else NAN, math.sqrt(sef / nf) if nf else NAN), n


def main() -> int:
    ap = argparse.ArgumentParser(description="P0-1 三方法评测")
    ap.add_argument("--factor", type=float, required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-ids", nargs="*", default=[])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--energies", type=float, nargs="*", default=[1.5, 3.0, 4.5, 6.0])
    ap.add_argument("--single-row", action="store_true",
                    help="Row-16-only 消融：预测时把输入投影的非 row16 行抹成零计数填充值，"
                         "必须与对应 run 的训练口径一致（B 组开启，A 组不开启）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-image", action="store_true")
    args = ap.parse_args()

    csv_p, cfg_p = resolve_csv(args.csv), ROOT / args.config
    rows_all = read_rows(csv_p)
    root = scan_root(rows_all)
    mean, std, fill = norm_of(load_cfg(cfg_p))
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[f={args.factor:g}] root={root.name} fill={fill} "
          f"norm=({mean:.4f},{std:.4f}) device={device}")

    out = {"factor": args.factor, "csv": args.csv, "config": args.config,
           "run_ids": args.run_ids, "data_root": str(root.relative_to(ROOT)),
           "norm": {"mean": mean, "std": std, "fill": fill}, "energies": {}}
    mask_fov = br.fov_mask()

    for energy in args.energies:
        ekey = f"{energy:g}"
        rows_e = [r for r in rows_all if abs(float(r["energy_MeV"]) - energy) < 1e-9]
        if not rows_e:
            continue
        print(f"\n=== {ekey} MeV ===", flush=True)
        L, ang, M = build_stack(rows_e, root, fill)
        ph_of = {int(float(r["target_angle_deg"])): r["phase"] for r in rows_e}
        split_of = {int(float(r["target_angle_deg"])): r["split"] for r in rows_e}
        dense = br.restrict_to_fov(br.recon_fbp(L[:, br.RECON_ROW, :], ang, "hamming"))

        cnn = {}
        for rid in args.run_ids:
            per_phase = {}
            for ph in ("A", "B"):
                per_phase[ph] = predict_phase(rid, cfg_p, csv_p, energy, ph,
                                              args.batch, device,
                                              single_row=args.single_row)
            cnn[rid] = per_phase
            print(f"  预测完成 {rid}", flush=True)

        Li, Lc = interp_linear(L), interp_cubic(L)

        agg: dict[str, dict[str, list]] = {}
        per_sample: list[dict] = []
        img_vals: dict[str, list] = {}

        for in_phase, missing in (("A", "B"), ("B", "A")):
            mis_idx = np.array([a for a in range(360) if ph_of[a] == missing])
            preds = {
                "linear_interp": Li[mis_idx],
                "cubic_interp": Lc[mis_idx],
            }
            for rid in args.run_ids:
                pr, prws = cnn[rid][missing]
                a_got = np.array([int(float(r["target_angle_deg"])) for r in prws])
                if not np.array_equal(a_got, mis_idx):
                    raise SystemExit(f"[FATAL] {rid} 预测角度与目标角度不对齐"
                                     f"（{a_got[:5]} vs {mis_idx[:5]}）")
                preds[rid] = pr
            tgt, msk = L[mis_idx], M[mis_idx]

            P = L.copy()
            P[mis_idx] = preds["linear_interp"]
            img_vals.setdefault("linear_interp", []).append(
                br.image_metrics(dense, br.restrict_to_fov(
                    br.recon_fbp(P[:, br.RECON_ROW, :], ang, "hamming")), mask_fov)["rmse_valid"])
            P = L.copy()
            P[mis_idx] = preds["cubic_interp"]
            img_vals.setdefault("cubic_interp", []).append(
                br.image_metrics(dense, br.restrict_to_fov(
                    br.recon_fbp(P[:, br.RECON_ROW, :], ang, "hamming")), mask_fov)["rmse_valid"])
            for rid in args.run_ids:
                P = L.copy()
                P[mis_idx] = preds[rid]
                img_vals.setdefault(rid, []).append(
                    br.image_metrics(dense, br.restrict_to_fov(
                        br.recon_fbp(P[:, br.RECON_ROW, :], ang, "hamming")), mask_fov)["rmse_valid"])

            for meth, pr in preds.items():
                r16 = rmse_pair(pr, tgt, msk, br.RECON_ROW)
                a32 = rmse_pair(pr, tgt, msk, None)
                slot = agg.setdefault(meth, {"row16": [], "all32": []})
                slot["row16"].append(r16[0][0])
                slot["all32"].append(a32[0][0])

            for k, a in enumerate(mis_idx):
                if split_of.get(int(a)) != "test":
                    continue
                m = M[a, br.RECON_ROW]
                if not m.any():
                    continue
                rowd = {"angle": int(a), "phase": missing, "n_valid": int(m.sum())}
                for meth, pr in preds.items():
                    rowd[meth] = float(np.sqrt(
                        (((pr[k, br.RECON_ROW] - L[a, br.RECON_ROW]) ** 2) * m).sum() / m.sum()))
                per_sample.append(rowd)

        rec = {meth: {"row16": float(np.mean(v["row16"])),
                      "all32": float(np.mean(v["all32"])),
                      "row16_by_phase": v["row16"],
                      "all32_by_phase": v["all32"]} for meth, v in agg.items()}
        imgd = {k: {"rmse_valid": float(np.mean(v)), "by_phase": v}
                for k, v in img_vals.items()}
        imgd["_dense_data_range"] = float(dense.max() - dense.min())

        out["energies"][ekey] = {"recon_domain": rec, "image_domain": imgd,
                                 "per_sample_test": per_sample, "n_test": len(per_sample)}
        for meth in ("linear_interp", "cubic_interp") + tuple(args.run_ids):
            if meth not in rec:
                continue
            print(f"    {meth:34s} row16={rec[meth]['row16']:.6f}  "
                  f"all32={rec[meth]['all32']:.6f}  "
                  f"image={imgd[meth]['rmse_valid']:.6f}", flush=True)

    out_p = Path(args.out) if args.out else RUNS / f"_dose_eval_f{args.factor:g}.json"
    out_p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] {out_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
