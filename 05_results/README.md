# 05_results —— 结果部分（总索引）

> 作业：**CISC3024 Pattern Recognition / AIassign2（AI Assignment #1）**
> 算法：**ConvNeXt**（*A ConvNet for the 2020s*, **CVPR 2022**, arXiv:2201.03545） ｜ 任务：**CIFAR-10 图像分类**
> **归属任务**：**T6（实验运行与结果收集）、T7（结果可视化与图表整理）** 的归档目录

本目录是作业的**结果部分**。为满足"**各实验独立存放、互不覆盖、互不交叉引用**"的要求，
每个实验各自占用一个**独立子目录**，其内部再按 **图表 / 日志 / 模型 / 指标** 四类组织。

---

## 1. 目录结构（实验隔离）

```
05_results/
├── README.md                        # 本文件：实验总索引
│
├── mvp_atto_subset5k/               # ── 实验 1（已完成，真实数据）────────────────
│   ├── README.md                    #    该实验的完整结果说明（含表 1–表 4）
│   ├── figures/                     #    【图表】
│   │   ├── figures_index.md         #      图表索引（T7 产物，含图 1–图 4 编号与说明）
│   │   ├── training_curves.png      #      图 1 训练曲线（损失/准确率/学习率）
│   │   ├── confusion_matrix.png     #      图 2 测试集混淆矩阵
│   │   ├── per_class_accuracy.png   #      图 3 逐类准确率
│   │   └── data_samples.png         #      图 4 数据样例
│   ├── logs/                        #    【日志】
│   │   ├── README.md                #      日志索引与执行阶段标注
│   │   ├── train.log                #      训练日志（带时间戳，12 个阶段）
│   │   ├── evaluate.log             #      独立评估日志
│   │   ├── history.json             #      逐轮结构化指标
│   │   └── history.csv              #      逐轮指标表（可直接作为报告表格）
│   ├── models/                      #    【模型】
│   │   ├── README.md                #      检查点结构、加载方式、架构与超参数
│   │   ├── best.pt                  #      验证集最优模型（epoch 10）
│   │   └── last.pt                  #      最后一轮模型（epoch 10）
│   └── metrics/                     #    【指标】
│       ├── README.md                #      字段含义说明
│       ├── config.json              #      超参数快照
│       ├── metrics.csv              #      一行式指标汇总表（T6 产物，可直接拼接成对比表）
│       ├── test_metrics.json        #      最终测试指标汇总
│       ├── per_class_metrics.json   #      逐类准确率 + 10×10 混淆矩阵
│       └── eval_metrics.json        #      独立评估（evaluate.py）结果
│
├── full_convnext_tiny_cifar10/      # ── 实验 2（已完成，本机 GPU + bf16，主结果）─────
│   ├── README.md                    #    结果总览（表 1–表 4 + 断点续训过程说明）
│   ├── figures/                     #    【图表】figures_index.md + 图 1–图 4
│   ├── logs/                        #    【日志】README.md + train.log / evaluate.log / history.*
│   ├── models/                      #    【模型】README.md + best.pt(epoch 59) / last.pt(epoch 60)
│   └── metrics/                     #    【指标】README.md + config.json / metrics.csv / *_metrics.json
│
├── convnext_cifar10_outputs.zip     #    原始运行包（Colab 端产物原样留存，含两张检查点）
│
├── _failed_runs/                    # ── 失败/无效运行（隔离存放，不作为报告结果）──
│   └── full_convnext_tiny_cifar10_amp_fp16_underflow/
│       ├── README.md                #    失败诊断报告（现象 + 证据链 + 根因 + 修复 + 复现）
│       ├── figures/ logs/ metrics/  #    该次运行的全部日志/指标/图表（权重文件留在 zip 内）
│
└── <后续实验目录>/                   # ── 由 --run-tag 生成，结构同上 ──────────────
                                      例如 cmp_*/、ablation_*/、seed4*/
```

> **隔离约定**：任何一次运行都必须通过 `--run-tag <name>` 指定独立的实验标签，
> 所有产物写入 `05_results/<name>/`。这样"全量训练 / 对比实验 / 消融实验 / 多种子重复"
> 彼此完全隔离，不会发生文件覆盖或交叉引用。

---

## 2. 实验清单

| 实验目录（`--run-tag`） | 目的 | 数据规模 | 设备 | 测试 Top-1 | 状态 |
|---|---|---|---|---|---|
| `mvp_atto_subset5k` | **I0 / MVP**：在 CPU 上端到端跑通全链路 | 训练 4,500 / 验证 500 / 测试 2,000 | CPU | **45.05 %** | ✅ 已完成 |
| `full_convnext_tiny_cifar10`（fp16 AMP 那次） | **I1（失败）**：全量数据 + 60 轮；fp16 梯度下溢导致骨干冻结（假收敛），结果无效 | 训练 45,000 / 验证 5,000 / 测试 10,000 | T4 / Colab | 25.73 %（无效） | ❌ 已归档至 `_failed_runs/` |
| `full_convnext_tiny_cifar10`（重跑，**主结果**） | **I1**：全量数据 + 60 轮，取得可写入报告的主结果 | 训练 45,000 / 验证 5,000 / 测试 10,000 | **本机 GPU RTX 4060 + bf16** | **92.66 %**（TTA **92.87 %**） | ✅ 已完成 |
| `cmp_*`（规划中） | **I2**：与基线模型对比 | 视配置 | GPU / Colab | — | ⏳ 待运行 |
| `ablation_*`（规划中） | **I2**：关键模块消融 | 视配置 | GPU / Colab | — | ⏳ 待运行 |
| `seed42/43/44`（规划中） | **I3**：多种子重复与鲁棒性 | 视配置 | GPU / Colab | — | ⏳ 待运行 |

> **写入报告的主结果**：`full_convnext_tiny_cifar10/` —— 真实全量 CIFAR-10、ConvNeXt-tiny、
> 60 轮、本机 RTX 4060（**bfloat16 + GradScaler**）训练 1 h 55 m，
> 测试集 **Top-1 = 92.66 %**（启用 TTA 后 **92.87 %**）、Top-5 = 99.56 %，宏平均 92.87 %，
> 最优验证 93.50 %（epoch 59）。完整结果说明见该目录的 `README.md`（表 1–表 4）。

---

## 3. 命名与任务编号映射（便于交付核对）

| 任务拆解约定的产物名 | 实际归档位置 | 说明 |
|---|---|---|
| `metrics.csv`（T6） | `05_results/<实验>/metrics/metrics.csv` | 一行一次实验的宽表，含逐类准确率列 |
| 运行日志 `*.log`（T6） | `05_results/<实验>/logs/` | `train.log`、`evaluate.log` |
| 模型文件 `*.pt`（T6） | `05_results/<实验>/models/` | `best.pt`、`last.pt` |
| 图表 `*.png`（T7） | `05_results/<实验>/figures/` | 4 张，编号见该目录的 `figures_index.md` |
| `figures_index.md`（T7） | `05_results/<实验>/figures/figures_index.md` | 图表索引（含图 1–图 4 编号与读法） |

> 除上述约定文件名外，每类目录另附 `README.md`（图表/日志/模型/指标的字段与读法说明），
> 以及实验自身的 `README.md`（结果总览）。

---

## 4. 三类结果与目录的对应关系

| 要求的分类 | 目录名 | 内含 |
|---|---|---|
| **图表** | `figures/` | 训练曲线、混淆矩阵、逐类准确率、数据样例 + `figures_index.md` |
| **日志** | `logs/` | `train.log`、`evaluate.log`、`history.json` + 索引说明 |
| **模型** | `models/` | `best.pt`、`last.pt` + 检查点结构与参数说明 |
| 指标（补充） | `metrics/` | `config.json`、`metrics.csv`、`test_metrics.json`、`per_class_metrics.json`、`eval_metrics.json` |

---

## 5. 代码 / 数据 / 结果三部分的引用关系

```
┌──────────────────────────────┐
│ 03_code（代码，T4）           │  全部由 AI 生成
│  configs.py / train.py        │
│  evaluate.py / src/**         │
└───────────┬──────────────────┘
            │  --data-root ../../04_data/cifar10   （读数据）
            │  --output-dir ../../05_results       （写结果根目录）
            │  --run-tag  <实验标签>                （实验隔离，决定子目录名）
            ▼
┌──────────────────────────────┐        ┌────────────────────────────────────────┐
│ 04_data（数据，T5）           │        │ 05_results（结果，T6/T7）               │
│  cifar10/   原始数据集        │───────▶│  <实验标签>/                            │
│  splits/    划分索引          │ 划分   │    figures/ 【图表】                    │
│  dataset_info.md  数据说明    │ 索引   │    logs/    【日志】                    │
└──────────────────────────────┘        │    models/  【模型】                    │
                                        │    metrics/ 【指标】                    │
                                        └────────────────────────────────────────┘
```

**可追溯链（从结果反查全过程）**：

| 想确认… | 查看 |
|---|---|
| 用了哪些超参数 | `05_results/<实验>/metrics/config.json`（与 `models/best.pt` 内 `config` 同源） |
| 数据来自哪里、划分如何 | `metrics/test_metrics.json` → `dataset.data_root`；`04_data/splits/split_info.json` |
| 数据规模是否为全量 | `metrics/test_metrics.json` → `dataset.is_full_scale` |
| 何时得到该指标 | `logs/train.log`（每行带 `[HH:MM:SS]` 时间戳） |
| 模型结构是否与论文一致 | `models/README.md` 第 3 节 + `logs/train.log` 的 `[model]` 结构摘要 |
| 指标如何计算 | `metrics/README.md` 第 2 节 + `03_code/src/engine/evaluator.py` |
| 多次实验如何横向比较 | 把各实验的 `metrics/metrics.csv` 纵向拼接（列顺序固定，可直接堆叠） |

---

## 6. 如何新增一个实验（保持隔离）

```bash
cd 03_code
# 每次运行都必须给一个唯一的实验标签；产物自动写入 05_results/<标签>/
# T4 上务必加 --no-amp（fp16 会让本模型浅层梯度下溢，见 03_code/README 第 12 节）
python train.py --run-tag full_convnext_tiny_cifar10 \
                --model tiny --epochs 60 --batch-size 128 --no-amp \
                --data-root ../04_data/cifar10 \
                --output-dir ../05_results
```

> 不带 `--run-tag` 时，产物会直接写入 `--output-dir` 根目录（旧行为），
> 仅建议在临时调试时使用，正式实验一律使用 `--run-tag` 以保证隔离。

---

## 7. 失败运行归档（`_failed_runs/`）

为保证"**只有可信结果进入正式实验目录**"，失败/无效的运行单独归档在 `_failed_runs/<标签>_<失败原因>/`，
**不与正式实验混放，也不参与结果汇总表**。归档内容 = 该次运行的日志/指标/图表（权重文件较大，保留在 zip 内）。

目前唯一一条：

| 归档目录 | 失败原因 | 关键判据 | 处置 |
|---|---|---|---|
| `_failed_runs/full_convnext_tiny_cifar10_amp_fp16_underflow/` | T4 上 fp16 AMP 的 GradScaler 动态 scale 崩塌（65536 → 0.0156），浅层 Block 梯度下溢为精确的 0，骨干网络被冻结（假收敛，测试 Top-1 = 25.73 %） | 检查点内 `scaler_state.scale = 0.0156`；`stages.1.*.pwconv1.weight` 的 `exp_avg_sq` 恒为 0；同权重在本地 fp32 复算得 25.71 %（与 Colab 一致，排除数据问题） | 已在代码层修复（bf16 优先 + `amp_scale` 监控 + 自动降级 fp32），并以 `--no-amp` 重跑 |

> 排查入口：该目录的 `README.md` 给出了完整证据链（现象表、逐层梯度统计、机理推导、反例对照）；
> 通用的 30 秒自查清单见 `03_code/README.md` 第 12.3 节。

---

## 8. 版本控制建议

模型检查点体积较大（`mvp_atto_subset5k` 的 `best.pt` / `last.pt` 各约 40 MB，
`full_convnext_tiny_cifar10` 的 tiny 检查点各约 334 MB，
`convnext_cifar10_outputs.zip` 约 350 MB），
若将工程托管至 Git 平台，建议排除二进制权重与运行包，仅保留日志、指标与图表：

```gitignore
05_results/*/models/*.pt
05_results/*/models/*.pth
05_results/_failed_runs/*/models/*.pt
05_results/*.zip
```
