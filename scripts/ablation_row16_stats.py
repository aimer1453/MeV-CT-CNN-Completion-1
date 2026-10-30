#!/usr/bin/env python
"""窄口径 Row-16 消融统计：只做**投影域**，并给出逐能量分解。

为什么单独写
------------
`p02_sweep_eval.py` 同时算图像域（FBP + PSNR + SSIM），占 90% 时间；而消融对照只需要
投影域 row16 / all32。本脚本复用它的 `EnergyStack` / `predict` / 掩膜 / 累加口径
（逐行同式），只跳过图像域，单 run 从 ~3.5 min 降到 ~25 s。

口径（与主表严格一致，不得改）
------------------------------
- step = 2（h = 1°），两个方向 c = 0/1（偶/奇相位），targ 恒取另一陪集
- 统计集：`sparse_targets.csv` 中 `split == "test"` 的 **45 个视角 × 4 档能量**
- 掩膜：row16 = `valid[θ, RECON_ROW]`，all32 = `valid[θ]`，`valid = (I>0)&(empty>0)`
- 池化：误差平方**跨像素跨能量求和后**再开方（不是先按能量求 RMSE 再平均）
- 归一化：从每个 run 的 `config.json → norm` 读取，**禁止**套主域常数

用法
----
    python scripts/ablation_row16_stats.py                     # 默认三配置九 run
    python scripts/ablation_row16_stats.py --runs <id> [...]   # 只算指定 run

产出
----
    runs/_ablation_row16_full.json   逐 run × 逐能量 × 逐方法的真值
    runs/_ablation_row16_table.md    三配置对照表（池化、逐能量、逐种子）
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import baseline_recon as br  # noqa: E402
import p02_sweep_eval as p2  # noqa: E402  复用其 EnergyStack / predict / 数据装载

STEP = 2
H = STEP // 2

CONFIGS: dict[str, list[str]] = {
    "masked": [
        "20261024_0400_cnn_bs8_lr1e-4",
        "20261024_1633_cnn_bs8_lr1e-4",
        "20261024_1805_cnn_bs8_lr1e-4",
    ],
    "unmasked": [
        "20261030_0319_cnn_bs8_lr1e-4_unmask_s22",
        "20261030_0352_cnn_bs8_lr1e-4_unmask_s23",
        "20261030_0426_cnn_bs8_lr1e-4_unmask_s24",
    ],
    "view2": [
        "20261026_0052_cnn_2view_bs8_lr1e-4",
        "20261030_0216_cnn_2view_bs8_lr1e-4_view2_s23",
        "20261030_0248_cnn_2view_bs8_lr1e-4_view2_s24",
    ],
}


def eval_run(rid: str) -> dict:
    cfg = json.loads((br.RUNS_DIR / rid / "config.json").read_text(encoding="utf-8"))
    norm = cfg["norm"]
    mean, std, fill = (float(norm["norm_mean"]), float(norm["norm_std"]),
                       float(norm["zero_log_fill"]))
    n_views = int(cfg.get("input_views", 3))

    files = p2.load_angles()
    split = p2.load_split()
    energies = list(p2.ENERGIES)
    stacks = {e: p2.EnergyStack(e, files[e], fill) for e in energies}

    import torch
    from models.unet import UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=n_views + 2).to(device)   # 两视图 = 4 通道，三视图 = 5
    ck = torch.load(br.RUNS_DIR / rid / "best.pt", map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    acc = {m: {r: {e: [0.0, 0] for e in energies} for r in ("row16", "all32")}
           for m in ("cnn", "linear", "cubic")}

    for c in range(STEP):
        for e in energies:
            st = stacks[e]
            targ = [a for a in range(360) if a % STEP == (c + H) % STEP]

            lin32 = np.empty((len(targ), 32, 1200), dtype=np.float64)
            cub32 = np.empty((len(targ), 32, 1200), dtype=np.float64)
            for i, th in enumerate(targ):
                lin32[i] = 0.5 * (st.L[(th - H) % 360] + st.L[(th + H) % 360])
                cub32[i] = (9.0 / 16.0) * (st.L[(th - H) % 360] + st.L[(th + H) % 360]) \
                    - (1.0 / 16.0) * (st.L[(th - 3 * H) % 360] + st.L[(th + 3 * H) % 360])
            cnn32 = p2.predict(model, device, [(e, th) for th in targ], stacks, files,
                               STEP, mean, std, fill, n_views=n_views).astype(np.float64)

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

    out: dict = {
        "run_id": rid,
        "seed": int(cfg["seed"]),
        "input_views": n_views,
        "loss_mode": cfg.get("loss_mode", "masked"),
        "epoch": int(ck.get("epoch", -1)),
        "n_test_views": 45,
        "per_energy": {},
        "pooled": {},
    }
    for meth in ("cnn", "linear", "cubic"):
        out["per_energy"][meth] = {}
        for roi in ("row16", "all32"):
            out["per_energy"][meth][roi] = {
                e: (math.sqrt(s / n) if n else float("nan"))
                for e, (s, n) in acc[meth][roi].items()}
            S = sum(v[0] for v in acc[meth][roi].values())
            N = sum(v[1] for v in acc[meth][roi].values())
            out["pooled"].setdefault(meth, {})[roi] = math.sqrt(S / N) if N else float("nan")
            out.setdefault("n_pixels", {})[roi] = int(N)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="只算这些 run（默认跑三配置九 run）")
    ap.add_argument("--out-json", default="runs/_ablation_row16_full.json")
    ap.add_argument("--out-md", default="runs/_ablation_row16_table.md")
    a = ap.parse_args(argv)

    todo = ({"custom": a.runs} if a.runs else CONFIGS)
    res: dict[str, list[dict]] = {}
    for cfg_name, rids in todo.items():
        res[cfg_name] = []
        for rid in rids:
            t0 = time.time()
            d = eval_run(rid)
            d["config"] = cfg_name
            res[cfg_name].append(d)
            print(f"[{cfg_name}] {rid}  row16={d['pooled']['cnn']['row16']:.6f} "
                  f"linear={d['pooled']['linear']['row16']:.6f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    Path(a.out_json).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                encoding="utf-8")

    # ---------------- 表
    L: list[str] = []
    append = L.append
    L.append("# Row-16 消融（masked / unmasked / 两视图）：投影域，45 test 视角，step=2")
    L.append("")
    L.append("口径：`sparse_targets.csv`、h=1°、两个方向、test split 45 视角 × 4 档能量；")
    L.append("掩膜 `(I>0)&(empty>0)`；**误差平方跨像素跨能量求和后再开方**（非先均值再平均）；")
    L.append("**评估恒为 masked**，与训练 loss 无关。生成：`scripts/ablation_row16_stats.py`。")
    L.append("")

    def mean_of(cfg_name: str, meth: str, roi: str, energy: str | None = None):
        vs = []
        for d in res.get(cfg_name, []):
            vs.append(d["pooled"][meth][roi] if energy is None
                      else d["per_energy"][meth][roi][energy])
        return vs

    L.append("## 1. 池化 Row-16（4 档能量合并，主口径）")
    L.append("")
    L.append("| 配置 | 种子 | CNN row16（逐种子） | 均值 ± SD | linear | **CNN/linear** |")
    L.append("|---|---|---|---|---|---|")
    base_masked = None
    for cfg_name in res:
        vs = mean_of(cfg_name, "cnn", "row16")
        mu, sd = float(np.mean(vs)), float(np.std(vs))
        lin = float(np.mean(mean_of(cfg_name, "linear", "row16")))
        L.append(f"| {cfg_name} | {len(vs)} | "
                 + ", ".join(f"{v:.6f}" for v in vs)
                 + f" | **{mu:.6f} ± {sd:.6f}** | {lin:.6f} | {mu / lin:.4f} |")
        if cfg_name == "masked":
            base_masked = mu
    append("")
    if base_masked:
        append("**相对 masked 的退化（逐种子配对，按同一种子配对，不跨种子混比）**：")
        append("")
        # 按 seed 配对：seed -> cnn row16
        pair: dict[str, dict[int, float]] = {}
        for cfg_name in res:
            pair[cfg_name] = {d["seed"]: d["pooled"]["cnn"]["row16"] for d in res[cfg_name]}
        base_pair = pair.get("masked", {})
        append("| 配置 | 逐种子配对增幅 | 均值 |")
        append("|---|---|---|")
        for cfg_name in res:
            rows_d = []
            for sd, v in pair[cfg_name].items():
                if sd in base_pair:
                    rows_d.append(v / base_pair[sd] - 1)
            if not rows_d:
                continue
            append(f"| {cfg_name} | " + " / ".join(f"{x:+.2%}" for x in rows_d)
                   + f" | **{float(np.mean(rows_d)):+.2%}** |")
        append("")

    L.append("## 2. 逐能量 Row-16（同口径，供核对「13.3% 从哪来」）")
    L.append("")
    L.append("| 配置 | 1.5 MeV | 3.0 MeV | 4.5 MeV | 6.0 MeV |")
    L.append("|---|---|---|---|---|")
    for cfg_name in res:
        cells = []
        for e in p2.ENERGIES:
            v = mean_of(cfg_name, "cnn", "row16", e)
            cells.append(f"{float(np.mean(v)):.6f}")
        L.append(f"| {cfg_name} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("对照 masked 的逐能量增幅：")
    L.append("")
    L.append("| 配置 | 1.5 MeV | 3.0 MeV | 4.5 MeV | 6.0 MeV |")
    L.append("|---|---|---|---|---|")
    for cfg_name in res:
        cells = []
        for e in p2.ENERGIES:
            m = float(np.mean(mean_of(cfg_name, "cnn", "row16", e)))
            b = float(np.mean(mean_of("masked", "cnn", "row16", e)))
            cells.append(f"{m / b - 1:+.2%}")
        L.append(f"| {cfg_name} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 3. all32（全 32 行）对照")
    L.append("")
    L.append("| 配置 | CNN all32 | linear all32 | CNN/linear | 相对 masked |")
    L.append("|---|---|---|---|---|")
    base_all = None
    for cfg_name in res:
        vs = mean_of(cfg_name, "cnn", "all32")
        mu = float(np.mean(vs))
        lin = float(np.mean(mean_of(cfg_name, "linear", "all32")))
        if cfg_name == "masked":
            base_all = mu
        L.append(f"| {cfg_name} | {mu:.6f} | {lin:.6f} | {mu / lin:.4f} | "
                 f"{(mu / base_all - 1):+.2%} |" if base_all else
                 f"| {cfg_name} | {mu:.6f} | {lin:.6f} | {mu / lin:.4f} | — |")
    L.append("")
    L.append("> 红线：all32 不可与 row16 混用。32 行里 31 行是非成像行，大量低通量像素由")
    L.append("> 正则/zero-fill 撑着，all32 会给出「CNN 全面领先」的相反印象。")
    L.append("")

    Path(a.out_md).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[OK] {a.out_json}\n[OK] {a.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
