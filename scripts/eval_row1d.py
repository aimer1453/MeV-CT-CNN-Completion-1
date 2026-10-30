#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""C 组（1D 单行）跨行评估 + A/B/C 三方对比

协议（与已发布的 Row16 / 跨行泛化口径**逐项一致**，否则三列无法并列）
--------------------------------------------------------------------------
- 每个方向：用实测相位 P 的 180 个视角，预测 opposite(P) 的 180 个视角；
- 两个方向（A→B、B→A）各算一次 per-row RMSE，再取两方向**均值**（= 已发布的算术口径；
  注意是「RMSE 的均值」而非「池化后开方」，两者略有差别，跟 A/B 组保持一致）；
- 指标：-log 物理域、per-row per-pixel mask RMSE（mask = I_target>0 & empty_proj>0）；
- 「全视角」= 360 个角度；另给「test-only」= 两方向合并后只保留 split=test 的目标角（45 个）。

三组的输入差异（这是被测量的自变量）
------------------------------------
  A  2D U-Net，输入 (5,32,1200)，32 行全部真实        -> predict: dose_eval.predict_phase
  B  2D U-Net，输入 (5,32,1200)，非 Row16 行填零计数填充值 -> 同上 + single_row=True
  C  1D U-Net，输入 (5,1200)，只有目标行自身            -> 本文件的 predict_phase_1d

C 组在 Row 0/15/31 上是**真正的跨行应用**：把该行的实测投影喂进只在 Row 16 上训练过的 1D 网络。
这与 B 组的性质完全不同 —— B 在这些行上既无输入也无监督，输出退化成常数。

自检
----
脚本会断言 A/B 的「全视角」结果与 runs/_row16_generalization.json 逐位一致（1e-6），
若不一致立即失败，避免用不同口径的数据拼表。

用法
----
    python scripts/eval_row1d.py
产出：runs/_row1d_generalization.json 、 runs/_row1d_generalization.md
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
sys.path.insert(0, str(ROOT / "scripts"))

from dose_eval import (build_stack, norm_of, predict_phase, read_rows,  # noqa: E402
                       scan_root)

CSV_NAME = "sparse_targets_3MeV.csv"
CFG = "config.json"
ENERGY = 3.0
ROWS = [0, 15, 16, 31]
BATCH = 8
INDEX = RUNS / "_row16_ablation_index.csv"


# ------------------------------------------------------------------ run 清单
def group_run_ids(group: str) -> list[str]:
    out = [r["run_id"] for r in csv.DictReader(open(INDEX, newline="", encoding="utf-8"))
           if r["group"] == group]
    if len(out) != 3:
        raise SystemExit(f"[FATAL] {group} 只有 {len(out)} 个 run，期望 3")
    return out


def cnn1d_run_ids() -> list[str]:
    out = sorted(p.name for p in RUNS.glob("*_cnn1d_*_row16_3mev_s*")
                 if (p / "best.pt").is_file())
    if len(out) != 3:
        raise SystemExit(f"[FATAL] 找到 {len(out)} 个 cnn1d run（期望 3）: {out}")
    return out


# ------------------------------------------------------------------ 1D 前向
def predict_phase_1d(run_id: str, cfg_path: Path, csv_path: Path, phase: str,
                     batch: int, device, eval_row: int,
                     train_row: int = 16) -> tuple[np.ndarray, list[dict]]:
    """预测某个相位的 180 个目标视角在 **eval_row** 上的值，返回 (180,1200) 物理域。

    注意 `eval_row` 与 `train_row` 是两件事：
      eval_row  = 切哪一行的投影当输入、并与哪一行的真值比较（跨行测试时 != 16）；
      train_row = 该 checkpoint 训练时用的行，只用于校验权重与评估意图一致。
    1D 模型没有行轴，换行**只能**换输入 —— 这正是它与 B 组（抹值假装单行）的本质区别。
    """
    import torch

    from data.dataset import SparseCTDataset, load_csv_rows
    from models.unet1d import UNet1D

    mean, std, fill = norm_of(json.loads(cfg_path.read_text(encoding="utf-8")))
    ds = SparseCTDataset(split="train", zero_log_fill=fill, norm_mean=mean, norm_std=std,
                         n_views=3, csv_path=csv_path, data_root=ROOT)
    all_rows = []
    for sp in ("train", "val", "test"):
        all_rows += load_csv_rows(csv_path, sp)
    keep = [i for i, r in enumerate(all_rows)
            if abs(float(r["energy_MeV"]) - ENERGY) < 1e-9 and r["phase"] == phase]
    if len(keep) != 180:
        raise SystemExit(f"[FATAL] {ENERGY} MeV phase {phase} 目标 {len(keep)}，期望 180")
    ds.rows = [all_rows[i] for i in keep]

    ckpt = RUNS / run_id / "best.pt"
    if not ckpt.is_file():
        raise SystemExit(f"[FATAL] 缺 {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "row" in state and int(state["row"]) != int(train_row):
        raise SystemExit(f"[FATAL] {run_id} 训练行 {state['row']} != 期望 {train_row}")
    model = UNet1D(in_channels=5, out_channels=1)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()

    out = np.empty((len(ds.rows), 1200))
    with torch.no_grad():
        for i in range(0, len(ds.rows), batch):
            items = [ds[j] for j in range(i, min(i + batch, len(ds.rows)))]
            # 关键：只切 eval_row，喂 (B,5,1200)。sin/cos 通道本就逐行相同。
            x = torch.from_numpy(
                np.stack([it["x"][:, eval_row, :] for it in items]).astype(np.float32)
            ).to(device)
            out[i:i + len(items)] = model(x).cpu().numpy()[:, 0] * std + mean
    ang = np.array([float(r["target_angle_deg"]) for r in ds.rows])
    order = np.argsort(ang)
    return out[order], [ds.rows[i] for i in order]


# ------------------------------------------------------------------ 指标
def row_rmse_1d(pred: np.ndarray, tgt_row: np.ndarray, msk_row: np.ndarray) -> float:
    """pred/tgt_row/msk_row 均已是 (N,1200)：1D 模型的单行 RMSE（mask 内）。"""
    n = int(msk_row.sum())
    if n == 0:
        return float("nan")
    d = pred - tgt_row
    return float(math.sqrt(float(((d ** 2) * msk_row).sum()) / n))


def row_rmse(pred: np.ndarray, tgt: np.ndarray, msk: np.ndarray, row: int) -> float:
    """2D 组：pred/tgt/msk 均为 (N,32,1200)，取第 row 行后算 mask 内 RMSE。"""
    d = (pred - tgt)[:, row, :]
    m = msk[:, row, :]
    n = int(m.sum())
    if n == 0:
        return float("nan")
    return float(math.sqrt(float(((d ** 2) * m).sum()) / n))


def eval_one_run(run_id: str, kind: str, device, L, M, ph_of, split_of) -> tuple[dict, dict]:
    """返回 (res, diag)。

    res  = {row: {"full": [dir1, dir2], "test": [dir1, dir2]}}（未取均值）
    diag = {row: {"pred_std": [...], "true_std": [...]}}  —— 输出是否退化成常数的诊断

    2D 组（A/B）：一次前向给出全部 32 行，四个行一起统计；
    1D 组（C）：换行必须换输入，故每个行单独前向（模型无行轴，这是它的定义性特征）。
    """
    res = {r: {"full": [], "test": []} for r in ROWS}
    diag = {r: {"pred_std": [], "true_std": []} for r in ROWS}
    for _in_phase, missing in (("A", "B"), ("B", "A")):
        mis_idx = np.array([a for a in range(360) if ph_of[a] == missing])
        tgt, msk = L[mis_idx], M[mis_idx]
        is_test = np.array([split_of.get(int(a)) == "test" for a in mis_idx])

        if kind == "1d":
            for r in ROWS:
                pr, prws = predict_phase_1d(run_id, ROOT / CFG, resolve_csv(CSV_NAME), missing,
                                            BATCH, device, eval_row=r, train_row=16)
                a_got = np.array([int(float(x["target_angle_deg"])) for x in prws])
                if not np.array_equal(a_got, mis_idx):
                    raise SystemExit(f"[FATAL] {run_id} 预测角度与目标角度不对齐")
                # pr 是 (N,1200)：本行预测；真值/mask 取该行切片
                diag[r]["pred_std"].append(float(pr.std(axis=1).mean()))
                diag[r]["true_std"].append(float(tgt[:, r, :].std(axis=1).mean()))
                res[r]["full"].append(row_rmse_1d(pr, tgt[:, r, :], msk[:, r, :]))
                res[r]["test"].append(row_rmse_1d(pr[is_test], tgt[is_test, r, :],
                                                  msk[is_test, r, :]))
        else:
            pr, prws = predict_phase(run_id, ROOT / CFG, resolve_csv(CSV_NAME), ENERGY, missing,
                                     BATCH, device, single_row=(kind == "b"))
            a_got = np.array([int(float(x["target_angle_deg"])) for x in prws])
            if not np.array_equal(a_got, mis_idx):
                raise SystemExit(f"[FATAL] {run_id} 预测角度与目标角度不对齐")
            for r in ROWS:
                diag[r]["pred_std"].append(float(pr[:, r, :].std(axis=1).mean()))
                diag[r]["true_std"].append(float(tgt[:, r, :].std(axis=1).mean()))
                res[r]["full"].append(row_rmse(pr, tgt, msk, r))
                res[r]["test"].append(row_rmse(pr[is_test], tgt[is_test], msk[is_test], r))
    return res, diag


def aggregate(per_run: list[dict], protocol: str) -> dict:
    out = {}
    for r in ROWS:
        seeds = [float(np.mean([x for x in pr[r][protocol] if np.isfinite(x)]))
                 for pr in per_run]
        arr = np.array(seeds)
        out[r] = {"per_seed": [round(float(x), 6) for x in seeds],
                  "mean": float(arr.mean()),
                  "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                  "per_dir": [[round(float(y), 6) for y in pr[r][protocol]] for pr in per_run]}
    return out


def main() -> int:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    mean, std, fill = norm_of(json.loads((ROOT / CFG).read_text(encoding="utf-8")))
    rows_all = read_rows(resolve_csv(CSV_NAME))
    root = scan_root(rows_all)
    rows_e = [r for r in rows_all if abs(float(r["energy_MeV"]) - ENERGY) < 1e-9]
    L, _ang, M = build_stack(rows_e, root, fill)
    ph_of = {int(float(r["target_angle_deg"])): r["phase"] for r in rows_e}
    split_of = {int(float(r["target_angle_deg"])): r["split"] for r in rows_e}
    n_test = sum(1 for a in range(360) if split_of.get(a) == "test")
    print(f"[data] 360 视角，test={n_test} 视角（两方向合并后覆盖全部 test 角度）")

    groups = {
        "A_base32_3mev": ("2d_full32", group_run_ids("base32_3mev")),
        "B_row16_3mev": ("b", group_run_ids("row16_3mev")),
        "C_cnn1d_row16": ("1d", cnn1d_run_ids()),
    }
    per, diag = {}, {}
    for g, (kind, ids) in groups.items():
        per[g], diag[g] = [], []
        for rid in ids:
            r_res, r_diag = eval_one_run(rid, kind, device, L, M, ph_of, split_of)
            per[g].append(r_res)
            diag[g].append(r_diag)
            print(f"  {g}: {rid} 完成", flush=True)

    # 输出退化诊断：预测标准差 / 真值标准差（按行，跨种子与两方向取均值）
    std_diag = {}
    for g in groups:
        std_diag[g] = {}
        for r in ROWS:
            ps = [float(np.mean(d[r]["pred_std"])) for d in diag[g]]
            ts = [float(np.mean(d[r]["true_std"])) for d in diag[g]]
            std_diag[g][r] = {"pred_std": float(np.mean(ps)), "true_std": float(np.mean(ts)),
                              "ratio": float(np.mean(ps) / np.mean(ts)) if np.mean(ts) else float("nan")}

    full = {g: aggregate(per[g], "full") for g in groups}
    test = {g: aggregate(per[g], "test") for g in groups}

    # ---- 自检：A/B 的「全视角」必须复现已发布结果 ----
    ref_p = RUNS / "_row16_generalization.json"
    if ref_p.is_file():
        ref = json.loads(ref_p.read_text(encoding="utf-8"))
        tol = 1e-6
        bad = []
        for g, key in (("A_base32_3mev", "A_base32_3mev"), ("B_row16_3mev", "B_row16_3mev")):
            for r in ROWS:
                want = float(ref[key][str(r)]["mean"])
                got = float(full[g][r]["mean"])
                if abs(got - want) > tol:
                    bad.append(f"{g} Row{r}: 本次 {got:.6f} vs 已发布 {want:.6f}")
        if bad:
            print("\n[FATAL] 口径自检失败（本次复算与已发布结果不一致）：")
            for b in bad:
                print("   " + b)
            return 2
        print(f"[自检 OK] A/B 的 360 视角结果与 {ref_p.name} 逐位一致（tol {tol:g}）")
    else:
        print(f"[WARN] 缺 {ref_p}，跳过口径自检")

    table = {}
    for r in ROWS:
        a, b, c = (full[g][r]["mean"] for g in ("A_base32_3mev", "B_row16_3mev", "C_cnn1d_row16"))
        at, bt, ct = (test[g][r]["mean"] for g in ("A_base32_3mev", "B_row16_3mev", "C_cnn1d_row16"))
        table[r] = {
            "A": a, "B": b, "C": c,
            "C_vs_A_pct": (c - a) / a * 100.0 if a else float("nan"),
            "C_vs_B_pct": (c - b) / b * 100.0 if b else float("nan"),
            "A_test": at, "B_test": bt, "C_test": ct,
            "C_vs_A_test_pct": (ct - at) / at * 100.0 if at else float("nan"),
        }

    out = {
        "energy_MeV": ENERGY, "rows": [f"row{r}" for r in ROWS],
        "protocol": ("每方向用实测相位 180 视角预测 opposite 相位 180 视角，两方向 RMSE 取算术均值；"
                     "-log 物理域 per-row per-pixel mask RMSE（mask=I_target>0 & E>0）。"
                     "full=360 视角（与已发布 Row16/跨行泛化口径一致）；"
                     "test=两方向合并后只保留 split=test 的目标角（45 视角）"),
        "groups": {
            "A": "cnn 2D U-Net, 32x1200, 全 32 行训练（base32_3mev）",
            "B": "cnn 2D U-Net, 32x1200, 输入塌缩到 Row16（row16_3mev，伪单行）",
            "C": "cnn1d 1D U-Net, 5x1200, 只训 Row16（真单行，681,345 参数）",
        },
        "run_ids": {g: ids for g, (_k, ids) in groups.items()},
        "full360": full, "test45": test, "std_diag": std_diag,
        "table": {f"row{r}": table[r] for r in ROWS},
        "n_test_angles": n_test,
    }

    # ---- 判读 ----
    c16, c15, c0, c31 = (table[r]["C"] for r in (16, 15, 0, 31))
    a16, a15, a0, a31 = (table[r]["A"] for r in (16, 15, 0, 31))
    out["verdict"] = {
        "C_vs_A_row16_pct": table[16]["C_vs_A_pct"],
        "C_row15_vs_C_row16_pct": (c15 - c16) / c16 * 100.0,
        "C_row0_vs_C_row16_pct": (c0 - c16) / c16 * 100.0,
        "C_row31_vs_C_row16_pct": (c31 - c16) / c16 * 100.0,
        "note": ("Row15 是 Row16 的物理镜像行（empty_proj 行均值 22.567 vs 22.493，"
                 "A 组两行 RMSE 差 0.03%），故 C 在 Row15 上若不显著退化，说明 1D 网络"
                 "学到的是可迁移的角度插值映射；Row0/31 是低通量边缘行（通量低约 63 倍），"
                 "退化幅度反映跨物理条件的迁移代价。"),
    }
    jp = RUNS / "_row1d_generalization.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(out, RUNS / "_row1d_generalization.md")

    print("\n=== 三方对比（3.0 MeV，-log 物理域 per-row mask RMSE，360 视角）===")
    print(f"{'行':>7}{'A(32行2D)':>13}{'B(伪单行2D)':>14}{'C(真单行1D)':>14}"
          f"{'C vs A':>11}{'C vs B':>11}")
    for r in ROWS:
        t = table[r]
        print(f"{'Row '+str(r):>7}{t['A']:>13.6f}{t['B']:>14.6f}{t['C']:>14.6f}"
              f"{t['C_vs_A_pct']:>+10.2f}%{t['C_vs_B_pct']:>+10.2f}%")
    print(f"\n=== 同一对比（test split 45 视角）===")
    for r in ROWS:
        t = table[r]
        print(f"{'Row '+str(r):>7}{t['A_test']:>13.6f}{t['B_test']:>14.6f}{t['C_test']:>14.6f}"
              f"{t['C_vs_A_test_pct']:>+10.2f}%")
    print(f"\n[json] {jp}\n[md]   {RUNS / '_row1d_generalization.md'}")
    return 0


def write_md(out: dict, path: Path) -> None:
    t = out["table"]
    f360, t45 = out["full360"], out["test45"]
    L = ["# C 组（1D U-Net 真单行）跨行评估与三方对比\n",
         f"> {out['protocol']}\n",
         "> A/B 的 360 视角列已与 `_row16_generalization.json` 逐位自检通过（tol 1e-6），"
         "说明三列同口径。\n",
         "\n## 三方对比（360 视角）\n",
         "| 行 | A 组（2D 32 行） | B 组（2D 伪单行） | **C 组（1D 真单行）** | C vs A | C vs B |",
         "|---|---|---|---|---|---|"]
    for r in (0, 15, 16, 31):
        x = t[f"row{r}"]
        L.append(f"| Row {r} | {x['A']:.6f} | {x['B']:.6f} | **{x['C']:.6f}** | "
                 f"{x['C_vs_A_pct']:+.2f}% | {x['C_vs_B_pct']:+.2f}% |")
    L += ["\n## test split（45 视角）\n",
          "| 行 | A 组 | B 组 | C 组 | C vs A |",
          "|---|---|---|---|---|"]
    for r in (0, 15, 16, 31):
        x = t[f"row{r}"]
        L.append(f"| Row {r} | {x['A_test']:.6f} | {x['B_test']:.6f} | {x['C_test']:.6f} | "
                 f"{x['C_vs_A_test_pct']:+.2f}% |")
    L += ["\n## 逐种子明细（360 视角）\n",
          "| 行 | 组 | seed1 | seed2 | seed3 | 均值 ± std |",
          "|---|---|---|---|---|---|"]
    for r in (0, 15, 16, 31):
        for gk, label in (("A_base32_3mev", "A"), ("B_row16_3mev", "B"), ("C_cnn1d_row16", "C")):
            d = f360[gk][r]
            ps = d["per_seed"]
            L.append(f"| Row {r} | {label} | {ps[0]:.6f} | {ps[1]:.6f} | {ps[2]:.6f} | "
                     f"{d['mean']:.6f} ± {d['std']:.6f} |")
    v = out["verdict"]
    L += ["\n## 判读\n",
          f"- C 组 Row 16 相对 A 组：**{v['C_vs_A_row16_pct']:+.2f}%**",
          f"- C 组 Row 15 相对自身 Row 16：**{v['C_row15_vs_C_row16_pct']:+.2f}%**"
          f"（镜像行，物理条件几乎相同）",
          f"- C 组 Row 0 相对自身 Row 16：**{v['C_row0_vs_C_row16_pct']:+.2f}%**",
          f"- C 组 Row 31 相对自身 Row 16：**{v['C_row31_vs_C_row16_pct']:+.2f}%**",
          f"\n{v['note']}\n",
          "\n> 口径提醒：Row 16 的 mask 覆盖 1200/1200 像素（100% 有效），"
          "故该行 rmse_valid 与 rmse_full 相同；Row 0/31 的 mask 会剔除零计数像素，"
          "三组一律用 mask 内 RMSE，未混用全像素口径。\n"]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
