#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/r34_angle_difficulty.py —— 逐角度"内在难度"体检（回答 R34 任务 5）

问题
----
contig270（270 训练行，test 315–359）几乎追平 random（0.0875 → 0.0897），
而 near45（90 训练行，test 90–134）崩到 0.1374。差异是否可能只是
"contig270 的 test 角刚好是容易的角度"？

做法（不依赖任何模型，纯数据属性）
--------------------------------
对**全部 360 个角**各算四个内在量（都是该角自身的性质，与 split 无关）：

  lin_rmse_row16 : 线性插值 0.5·(L[θ−1]+L[θ+1]) 在 row16 上的 masked RMSE
                   —— 这是"任何方法在该角的最好参照"，也是 test 难度的直接度量
  ang_grad_row16 : row16 上 |L[θ+1]−L[θ−1]| 的均值
                   —— 该角需要被"补"多少角度变化，越大越难
  mean_signal    : row16 上 L[θ] 的有效像素均值（通量水平）
  n_valid_row16  : row16 有效像素数

然后把每套 split 的 test 角集合拿去和**随机同规模抽样**比：
实际 test 集的均值落在 20000 次随机抽样的第几百分位。
若接近 50 → test 集难度与随机无异，"容易角度"解释不成立。

用法
----
    python scripts/r34_angle_difficulty.py            # 3.0 MeV
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.csv_paths import resolve_csv  # noqa: E402
import baseline_recon as br              # noqa: E402
import p02_sweep_eval as swe             # noqa: E402

ENERGY = "3"
SPLIT_CSVS = {
    "random": "csv_data/sparse_targets_3MeV.csv",
    "contig270": "csv_data/sparse_targets_3MeV_contig270.csv",
    "near45": "csv_data/sparse_targets_3MeV_near45.csv",
    "rand90": "csv_data/sparse_targets_3MeV_rand90.csv",
    "even90": "csv_data/sparse_targets_3MeV_even90.csv",
    "grouped": "csv_data/sparse_targets_3MeV_grouped.csv",
}
N_DRAW = 20000
RNG_SEED = 20261022


def split_of(csv_path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[int(round(float(r["target_angle_deg"])))] = r["split"]
    return out


def main() -> int:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    fill = float(cfg["norm"]["zero_log_fill"])
    files = swe.load_angles()[ENERGY]
    st = swe.EnergyStack(ENERGY, files, fill)
    row = br.RECON_ROW
    print(f"[load] {ENERGY} MeV  360 views  row={row}", flush=True)

    per_angle: dict[int, dict] = {}
    for th in range(360):
        m1, p1 = (th - 1) % 360, (th + 1) % 360
        gt = st.row16[th]
        mk = st.valid[th, row, :]
        pred = 0.5 * (st.row16[m1] + st.row16[p1])
        rmse = float(np.sqrt(np.mean((pred[mk] - gt[mk]) ** 2))) if mk.any() else float("nan")
        grad = st.row16[p1] - st.row16[m1]
        per_angle[th] = {
            "lin_rmse_row16": rmse,
            "ang_grad_row16": float(np.mean(np.abs(grad[mk]))) if mk.any() else float("nan"),
            "abs_grad_row16": float(np.sqrt(np.mean(grad[mk] ** 2))) if mk.any() else float("nan"),
            "mean_signal_row16": float(np.mean(gt[mk])) if mk.any() else float("nan"),
            "n_valid_row16": int(mk.sum()),
        }

    lin = np.array([per_angle[a]["lin_rmse_row16"] for a in range(360)])
    grad = np.array([per_angle[a]["ang_grad_row16"] for a in range(360)])
    print(f"[global] lin_rmse_row16  mean={lin.mean():.6f}  sd={lin.std():.6f}  "
          f"min={lin.min():.6f}@{lin.argmin()}  max={lin.max():.6f}@{lin.argmax()}", flush=True)

    rng = np.random.default_rng(RNG_SEED)
    report: dict[str, dict] = {}
    for name, rel in SPLIT_CSVS.items():
        sp = split_of(resolve_csv(rel))
        test = np.array(sorted(a for a, v in sp.items() if v == "test"))
        train = set(a for a, v in sp.items() if v == "train")
        n = len(test)
        if n == 0:
            continue
        draws = np.stack([lin[rng.choice(360, n, replace=False)] for _ in range(N_DRAW)])
        d_mean = draws.mean(axis=1)
        d_grad = np.stack([grad[rng.choice(360, n, replace=False)] for _ in range(N_DRAW)]).mean(axis=1)

        # 覆盖度：test 角到最近 train 角的环形距离；θ±1 是否都在 train（bracket）
        tr = np.array(sorted(train))
        d = np.abs(test[:, None] - tr[None, :])
        d = np.minimum(d, 360 - d)
        nn = d.min(axis=1)
        brack = np.mean([((a - 1) % 360 in train) and ((a + 1) % 360 in train) for a in test])

        actual = float(lin[test].mean())
        pct = float((d_mean < actual).mean() * 100.0)
        g_actual = float(grad[test].mean())
        g_pct = float((d_grad < g_actual).mean() * 100.0)
        report[name] = {
            "csv": rel, "n_test": int(n),
            "test_angle_range": [int(test.min()), int(test.max())],
            "lin_rmse_row16_test_mean": actual,
            "lin_rmse_difficulty_percentile": pct,
            "ang_grad_test_mean": g_actual,
            "ang_grad_percentile": g_pct,
            "nn_train_dist_median": float(np.median(nn)),
            "nn_train_dist_max": float(nn.max()),
            "bracket_rate": float(brack),
            "n_train": len(train),
        }
        print(f"[{name:10s}] n_test={n:3d}  angles {test.min():3d}-{test.max():3d}  "
              f"lin={actual:.6f} (P{pct:4.1f})  grad={g_actual:.5f} (P{g_pct:4.1f})  "
              f"nn_train median={np.median(nn):5.1f}° max={nn.max():5.1f}°  "
              f"bracket={brack*100:5.1f}%", flush=True)

    out = {"energy_MeV": float(ENERGY), "recon_row": int(row),
           "n_draw": N_DRAW, "rng_seed": RNG_SEED,
           "global": {
               "lin_rmse_row16_mean": float(lin.mean()),
               "lin_rmse_row16_sd": float(lin.std()),
               "lin_rmse_row16_p05": float(np.percentile(lin, 5)),
               "lin_rmse_row16_p95": float(np.percentile(lin, 95)),
               "ang_grad_mean": float(grad.mean()),
           },
           "splits": report,
           "per_angle": {str(a): per_angle[a] for a in range(360)}}
    p = br.RUNS_DIR / "_r34_angle_difficulty.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[json] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
