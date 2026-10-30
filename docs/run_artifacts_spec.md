# 实验记录规范（runs/ 归档标准）

> 状态：生效中。落库日期 2026-10-22。
> 目的：为后续对比图与论文准备可追溯、可并列比较的实验记录。
> 约束级别：**强制**。任何训练/评估运行未按本规范落盘，视为该次运行无效。

## 1. 目录命名

每次训练 / 评估运行，在 `runs/` 下新建**独立**目录，一次性、不复用、不覆盖：

```
runs/<YYYYMMDD_HHMM>_<method>_bs<batch>_lr<lr>/
```

示例：

```
runs/20261022_1430_cnn_bs8_lr1e-4/
runs/20261022_1530_fbp/
runs/20261022_1540_sart/
runs/20261022_1550_linear_interp/
```

- `<method>`：`cnn` / `fbp` / `sart` / `linear_interp`（短横线用下划线）
- `<batch>` / `<lr>`：评估型基线（fbp/sart/linear_interp）无此参数时，省略对应片段，只保留 `_<method>`
- 时间戳取运行**启动**时刻，本地时区

## 2. 目录内必备文件

| 文件 | 内容 | 必须 |
|---|---|---|
| `config.json` | 完整参数快照 | 是 |
| `train_log.csv` | 每 epoch 一行（基线方法为每档一行，`epoch` 填 `-1`） | 是 |
| `test_metrics.json` | 最终测试指标 | 是 |

权重（`*.pt` / `*.pth`）、大数组（`*.npy`）、图片（`*.png`）可留在同目录，**但一律不进 Git**。

## 3. `config.json` 字段

必须包含（缺项即视为记录不完整）：

```json
{
  "run_id": "20261022_1430_cnn_bs8_lr1e-4",
  "method": "cnn",
  "git_commit": "<40 位 hash>",
  "git_dirty": false,
  "seed": 20261022,
  "batch": 8,
  "lr": 1e-4,
  "epochs": 5,
  "warmup": 50,
  "optimizer": "adam",
  "norm": {
    "norm_mean": 0.7202,
    "norm_std": 0.9249,
    "norm_scope": "global",
    "zero_log_fill": 5.5,
    "zero_fill_normalized": 5.1678,
    "zero_fill_normalized_derivation": "(zero_log_fill - norm_mean) / norm_std = (5.50 - 0.7202) / 0.9249 = 5.1679；字段按裁决书写 5.1678（截断差 1e-4），运行时以现算为准",
    "zero_fill_normalized_usage": "仅供人工核对；运行时代码必须现算，禁止读该字段当输入",
    "domain": "-log((I+1e-10)/(E+1e-10))",
    "fill_order_note": "填充发生在归一化之前的 -log 域；归一化域填充值由上式算出，不是常数 5.5",
    "recomputed": true,
    "source": "新电脑 sparse_targets.csv split=train 的 1080 个 target，排除 (I<=0)|(E<=0) 像素后统计；val/test 直接复用",
    "norm_per_energy_reference": {
      "_note": "仅参考，不用于训练",
      "1.5": {"mean": 0.6315, "std": 1.18, "zero_log_fill": 6.3},
      "3": {"mean": 0.6891, "std": 0.93, "zero_log_fill": 5.6},
      "4.5": {"mean": 0.7547, "std": 0.8104, "zero_log_fill": 4.9},
      "6": {"mean": 0.7875, "std": 0.7729, "zero_log_fill": 4.5}
    }
  },
  "coverage_definition": {
    "coverage": "I_target > 0 像素数 / 总像素数（资产口径）",
    "valid_coverage": "(I_target > 0) & (empty_proj > 0) 像素数 / 总像素数（评估口径）",
    "rmse_valid_uses": "mask = (I_target > 0) & (empty_proj > 0)"
  },
  "mask": {
    "per_pixel": true,
    "zero_count_excluded_from_loss": true,
    "eps": "<实际值>"
  },
  "data": {
    "manifest": "train_manifest_first_run.csv",
    "targets": "sparse_targets.csv",
    "angles": "angles.csv",
    "energies_MeV": [1.5, 3.0, 4.5, 6.0],
    "split_sizes": {"train": 1080, "val": 180, "test": 180}
  },
  "env": {
    "python": "<版本>",
    "torch": "<版本>",
    "cuda": "<版本或 null>",
    "gpu": "<型号或 CPU>"
  }
}
```

## 4. `train_log.csv` 字段（顺序固定，14 列）

```
epoch,split,energy_MeV,n_samples,coverage,valid_coverage,rmse_valid,rmse_full,psnr,ssim,n_invariant_warn,lr,loss_train,loss_val
```

- `epoch`：训练轮次，从 0 或 1 起（在 config 里注明起算点）；基线方法填 `-1`
- `split`：`train` / `val` / `test`
- `energy_MeV`：`1.5` / `3` / `4.5` / `6`（分能量分别记录，禁止只写汇总行）
- `n_samples`：该行统计的样本数
- `coverage`：**资产口径**，`I_target > 0` 像素数 / 总像素数
- `valid_coverage`：**评估口径**，`(I_target > 0) & (empty_proj > 0)` 像素数 / 总像素数，与 `mask` 完全一致
  - 恒有 `coverage >= valid_coverage`；两者定义见 `config.json` 的 `coverage_definition`，**禁止混用、禁止只记一个**
- `rmse_valid`：有效像素（非零计数）上的物理域 RMSE —— 主指标
- `rmse_full`：全像素物理域 RMSE
- `psnr` / `ssim`：在该行统计口径下的值，未计算填 `NaN`
- `n_invariant_warn`：该行（该 split × 该能量）中，逐样本 `rmse_full < rmse_valid` 的样本数。
  逐样本违反**只告警不中断**；首个告警样本的 `idx / energy / target_angle_deg / rmse_valid / rmse_full`
  打印在 stdout。若某 epoch 的告警比例（train+val 合计）超过 **5%**，训练停止并上报等裁决
- `lr`：该 epoch 的学习率（基线填 `NaN`）
- `loss_train` / `loss_val`：基线填 `NaN`

**硬性关系（分两层）**：

- **行级（分能量池化）**：同一行必须满足 `rmse_full >= rmse_valid`（方向不能反），违反即 `TrainSpecError` 终止，判该次运行无效
- **overall 池化**：跨能量按像素加权池化后同样必须满足，违反即终止
- **逐样本**：`rmse_full < rmse_valid` 只计入 `n_invariant_warn` 并打印首个样本，**不中断训练**

### 4.1 `test_per_sample.csv` 字段（仅 CNN，11 列）

test 阶段逐样本落盘，路径 `runs/<run_id>/test_per_sample.csv`：

```
energy_MeV,target_angle_deg,split,n_valid_pixels,coverage,valid_coverage,rmse_valid,rmse_full,psnr,ssim,rmse_invariant_warn
```

| 字段 | 含义 |
|---|---|
| `energy_MeV` | `1.5` / `3` / `4.5` / `6` |
| `target_angle_deg` | 该样本的 target 角度（来自 `sparse_targets.csv`，不做运行时推导） |
| `split` | 固定为 `test` |
| `n_valid_pixels` | `mask` 为真的像素数 |
| `coverage` | 资产口径（`I_target > 0` / 总量） |
| `valid_coverage` | 评估口径（`(I>0)&(E>0)` / 总量），与 `mask.mean()` 一致 |
| `rmse_valid` | 有效像素物理域 RMSE |
| `rmse_full` | 全像素物理域 RMSE |
| `psnr` / `ssim` | 口径待定，当前一律 `NaN` |
| `rmse_invariant_warn` | 布尔。`True` 表示该样本 `rmse_full < rmse_valid`（已计入告警，不影响落盘） |

## 5. `test_metrics.json` 结构

```json
{
  "run_id": "...",
  "git_commit": "...",
  "overall": {"n_samples": 180, "rmse_valid": 0.0, "rmse_full": 0.0, "psnr": 0.0, "ssim": 0.0},
  "per_energy": {
    "1.5": {"n_samples": 45, "coverage": 0.0, "valid_coverage": 0.0, "rmse_valid": 0.0, "rmse_full": 0.0, "psnr": 0.0, "ssim": 0.0},
    "3":   {"...": "同上"},
    "4.5": {"...": "同上"},
    "6":   {"...": "同上"}
  }
}
```

## 6. 基线方法并列

`fbp` / `sart` / `linear_interp` 的评估结果**必须**按上述完全相同的目录结构与字段格式存入 `runs/`，使后续对比图可以无差别读取：

- `config.json`：`method` 填对应值，`optimizer`/`batch`/`lr`/`warmup` 填 `null`
- `train_log.csv`：每档一行（`epoch=-1`），`lr`/`loss_*` 填 `NaN`
- `test_metrics.json`：结构与 CNN 完全一致

## 7. 裁决结果

> 本节 C1–C3 为当前生效口径；`runs/` 与 Git 的执行细则见文末附录。

### C1 全局归一化 —— **已裁决：全局一套，分档仅参考**

- `norm_scope = "global"`，实际配置值：`norm_mean = 0.7202`、`norm_std = 0.9249`、`zero_log_fill = 5.50`
- `zero_fill_normalized = (5.50 - 0.7202) / 0.9249 = 5.1679`；字段里按裁决书写 5.1678（截断差 1e-4），**运行时一律现算，代码禁止写死**
- `norm_per_energy_reference` 存分档数值，**仅参考、不用于训练**：

  | energy_MeV | mean | std | zero_log_fill |
  |---|---|---|---|
  | 1.5 | 0.6315 | 1.1800 | 6.30 |
  | 3 | 0.6891 | 0.9300 | 5.60 |
  | 4.5 | 0.7547 | 0.8104 | 4.90 |
  | 6 | 0.7875 | 0.7729 | 4.50 |

- **全局归一化的已知代价**：归一化后 1.5 MeV 的 std 约 **1.28**（1.1800/0.9249=1.2758）、6 MeV 约 **0.84**（0.7729/0.9249=0.8357）
- **触发重议条件**：若 5 epoch 后 1.5 MeV 的 `val_rmse_phys` 异常，再提交分档方案等裁决；在此之前不得擅自改 `norm_scope`
- 旧机器统计量 `0.370 / 2.726` 已作废（四种 `-log` 口径下均超阈：Δmean −0.44 ~ −15.03，Δstd −19% ~ −76%）

### C2 `coverage` 的定义 —— **已固定：两个字段，禁止混用**

| 字段 | 定义 | 用途 |
|---|---|---|
| `coverage` | `I_target > 0` 像素数 / 总像素数 | 资产口径 |
| `valid_coverage` | `(I_target > 0) & (empty_proj > 0)` 像素数 / 总像素数 | 评估口径，与 `mask` 完全一致 |

- 恒有 `coverage >= valid_coverage`（后者条件更严）；`dataset.py` 对此有运行时断言，违反直接抛 `DatasetSpecError`
- `config.json` 的 `coverage_definition` 写成三段式字典，分别说明两个口径与 `rmse_valid_uses`
- 当前 4 档实测（`split=train` 全部 1080 个 target，逐样本求均值）：

  | energy_MeV | 1.5 | 3 | 4.5 | 6 | ALL |
  |---|---|---|---|---|---|
  | `coverage`（资产） | 0.8259 | 0.9607 | 0.9903 | 0.9978 | 0.9437 |
  | `valid_coverage`（评估） | 0.8053 | 0.9573 | 0.9901 | 0.9978 | 0.9376 |
  | 差值 | 0.0206 | 0.0034 | 0.0002 | 0.0000 | 0.0061 |

  - 1080 个样本逐条断言 `coverage >= valid_coverage`，违例 **0**
  - 单样本最小值：`coverage` 0.8186，`valid_coverage` 0.7998
  - **勘误**：早期报告的"1.5MeV 非零占比 78.9%"系统计脚本标注错误（把 `零/非零` 当成了 `零/总量`），真实值为 **82.59%**，与此处及阈值扫描结果一致。该 78.9% 已作废
- `empty_proj <= 0` 的像素占比实测：1.5MeV **4.70%**、3MeV 0.61%、4.5MeV 0.04%、6MeV 0.00% → 两口径差值主要由 1.5MeV 贡献
- **禁止在任何日志 / config / README 中写 34%**，该数字已作废

### C3 norm 统计量 —— **已对齐**

- 统计来源：**新电脑** `sparse_targets.csv` 中 `split=train` 的 1080 个 target 文件，不使用旧机器任何数值
- 计算域：`-log((I + 1e-10) / (E + 1e-10))`，`E` 为同能量目录的 `empty_proj`
- 排除条件：`(I <= 0) | (E <= 0)` 的像素不参与统计
- 样本规模：41,472,000 像素 = 有效 38,884,021 + 无效 2,587,979
- `ZERO_LOG_FILL = ceil_to_0.1(valid 的 p99.9) = ceil_to_0.1(5.4275) = 5.50`
- 全局 valid 口径：`P1 = -1.222`、`P50 = 0.577`、`P99 = 3.443`、`mean = 0.7202`、`std = 0.9249`
- 交叉校验：输入文件并集（`ch_prev/ch0/ch_next` 去重 1422 个）的 mean/std 与 1080 个 target 对比，最大 |Δmean| = 0.0018、最大 |Δstd%| = 0.24%，**远低于 0.05 / 5% 阈值**
- 完整数值见 `docs/norm_stats_proposal.json`，分布图见 `docs/figs/`（`scripts/compute_norm_stats.py` + `scripts/plot_norm_stats.py` 可复跑）

---

## 附录：`runs/` 与 Git 的关系 —— B 方案（执行细则）

`.gitignore` 已按 B 方案修改（2026-10-22）：

- **可入库**：`runs/**/config.json`、`runs/**/train_log.csv`、`runs/**/test_metrics.json`、`runs/**/norm_stats.json`
- **被忽略**：`runs/**/*.pt`、`runs/**/*.pth`、`runs/**/*.ckpt`、`runs/**/*.npy`、`runs/**/*.png`、`runs/**/*.jpg`
- 同时移除了全局的 `train_log.csv`、`test_metrics.json` 两条忽略规则，避免与入库要求冲突
- 验证方式：`git check-ignore -v <path>` 逐项确认（见提交记录）

---

**硬约束**：后续 config 生成脚本在 norm 字段缺失时必须报错，不得填默认值；`zero_fill_normalized` 必须由 `(zero_log_fill - norm_mean) / norm_std` 现算，不得读取已有字段。
