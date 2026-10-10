# mvp_atto_subset5k —— MVP 小规模实验结果

> **实验标签**：`mvp_atto_subset5k` ｜ **对应迭代**：**I0（MVP 端到端打通）**
> **归属任务**：**T6（实验运行与结果收集）、T7（结果可视化与图表整理）**
> 算法：**ConvNeXt**（*A ConvNet for the 2020s*, CVPR 2022, arXiv:2201.03545） ｜ 任务：CIFAR-10 图像分类
> 实验日期：2026-10-09 ｜ 全部内容为 `03_code` 在真实数据上**实际运行产生**

---

## 1. 目录结构（本实验自包含）

```
mvp_atto_subset5k/
├── README.md                        # 本文件：结果总览（表 1–表 4）
├── figures/                         # 【图表】
│   ├── figures_index.md             #   图表索引：图 1–图 4 的编号、含义、读法（T7 产物）
│   ├── training_curves.png          #   图 1：损失曲线 / 准确率曲线 / 学习率曲线
│   ├── confusion_matrix.png         #   图 2：测试集混淆矩阵（按真实类别归一化）
│   ├── per_class_accuracy.png       #   图 3：逐类准确率条形图
│   └── data_samples.png             #   图 4：CIFAR-10 数据样例网格
├── logs/                            # 【日志】
│   ├── README.md                    #   日志索引与 12 个执行阶段标注
│   ├── train.log                    #   训练日志（带 [HH:MM:SS] 时间戳）
│   ├── evaluate.log                 #   独立评估日志
│   ├── history.json                 #   逐轮结构化指标（loss / acc / lr / 耗时）
│   └── history.csv                  #   逐轮指标表（与表 4 同源，可直接进报告）
├── models/                          # 【模型】
│   ├── README.md                    #   检查点结构、加载示例、架构与超参数说明
│   ├── best.pt                      #   验证集最优模型（epoch 10，约 40 MB）
│   └── last.pt                      #   最后一轮模型（epoch 10，约 40 MB）
└── metrics/                         # 【指标】
    ├── README.md                    #   各字段含义说明
    ├── config.json                  #   本次运行的全部超参数快照
    ├── metrics.csv                  #   一行式指标汇总表（40 列，含逐类准确率，T6 产物）
    ├── test_metrics.json            #   最终测试指标汇总
    ├── per_class_metrics.json       #   逐类准确率 + 10×10 混淆矩阵
    └── eval_metrics.json            #   独立评估（evaluate.py）结果
```

> **关于目录整理**：本实验目录由阶段 A 的「实验目录隔离」整理而成
> （原先直接位于 `05_results/` 根下）。`logs/` 中日志文本内记录的路径字符串
> （如 `../05_results\figures\...`）反映的是**整理前**的布局；
> 为保证日志作为过程证据的**原始性**，未对其做任何改写。本实验目录内部的
> `README.md`、`figures/`、`logs/`、`models/`、`metrics/` 均已自包含，
> 不依赖其它实验目录。

---

## 2. 表 1 —— 实验设置

| 项目 | 值 |
|---|---|
| 实验标签 | `mvp_atto_subset5k` |
| 算法 / 类别 | ConvNeXt ／ Deep CNN |
| 模型预设 | `atto`（阶段通道 40/80/160/320，Block 数 2/2/6/2） |
| 参数量 | 3,376,890（3.38 M） |
| Stem 类型 | `conv`（3×3 stride=1，适配 32×32 输入） |
| 数据集 | CIFAR-10（`../../04_data/cifar10`） |
| **数据规模** | **训练 4,500 / 验证 500 / 测试 2,000（快速验证子集，`is_full_scale = false`）** |
| 优化器 | AdamW（lr=2e-3，weight decay=0.05，1 维参数豁免权重衰减） |
| 学习率调度 | 线性 warmup 2 轮 + 余弦退火至 1e-5 |
| 批大小 | 128（等效 batch 128，无梯度累积） |
| 训练轮数 | 10 |
| 数据增强 | RandomCrop(32, padding=4, reflect) + RandomHorizontalFlip + RandomErasing(p=0.25)；关闭 Mixup/CutMix |
| 标签平滑 | 0.1 |
| 随机种子 | 42 |
| 计算设备 | **CPU**（20 逻辑核，PyTorch 14 线程） |
| 总耗时 | **18 分 41 秒** |

> ⚠️ **重要**：本次存档的是**小规模 MVP 实验**（仅 4,500 张训练样本、10 轮），
> 目的是在有限算力下**完整跑通并验证整条流水线**（对应执行计划的迭代 I0）。
> 其准确率**不能**与全量数据结果或文献结果直接比较；
> 全量实验（`--model tiny --epochs 60`）请在 GPU / Colab 上运行，预期 Top-1 为 93–95 %。

---

## 3. 表 2 —— 测试集总体指标

基于 `models/best.pt`，测试集 2,000 张：

| 指标 | 数值 |
|---|---|
| 测试损失（交叉熵，含标签平滑 0.1） | **1.6696** |
| **Top-1 准确率** | **45.05 %** |
| Top-5 准确率 | **89.95 %** |
| 宏平均准确率（各类准确率均值） | 44.78 % |
| 正确样本数 / 总样本数 | 901 / 2,000 |
| 最优验证准确率（模型选择依据） | 45.20 %（epoch 10） |

> 随机基线：10 类均匀猜测 Top-1 = 10 %、Top-5 = 50 %。
> 本实验分别高出 **35.05 / 39.95 个百分点**，证明流水线有效。

---

## 4. 表 3 —— 逐类准确率（测试集）

| 类别 | 准确率 | 支撑样本 | 类别 | 准确率 | 支撑样本 |
|---|---|---|---|---|---|
| airplane | 54.24 % | 177 | dog | 32.16 % | 199 |
| automobile | 57.47 % | 221 | frog | 48.40 % | 188 |
| bird | 24.34 % | 189 | horse | 59.09 % | 198 |
| cat | 28.37 % | 208 | ship | **61.50 %** | 213 |
| deer | 27.32 % | 194 | truck | 54.93 % | 213 |

* **最好**：ship（61.50 %）、horse（59.09 %）、automobile（57.47 %）—— 类内形态一致、背景线索强。
* **最差**：bird（24.34 %）、deer（27.32 %）、cat（28.37 %）—— 与 `airplane / deer / dog` 易混淆，
  且在每类仅约 450 张训练样本的条件下更难学习。
* 该结论与 `figures/confusion_matrix.png`（图 2）中的高亮非对角区域一致。

---

## 5. 表 4 —— 逐轮训练指标（源自 `logs/history.json`）

| Epoch | train_loss | train_acc | val_loss | val_acc | 学习率 | 训练耗时 |
|---|---|---|---|---|---|---|
| 1 | 2.1323 | 22.71 % | 2.0975 | 27.00 % | 1.000e-03 | 124.3 s |
| 2 | 2.0701 | 25.93 % | 2.0574 | 25.60 % | 2.000e-03 | 94.7 s |
| 3 | 1.9993 | 29.02 % | 1.9912 | 32.20 % | 2.000e-03 | 101.9 s |
| 4 | 1.9602 | 30.84 % | 1.9618 | 30.60 % | 1.924e-03 | 82.3 s |
| 5 | 1.8941 | 34.82 % | 1.9328 | 34.40 % | 1.709e-03 | 72.3 s |
| 6 | 1.8351 | 38.84 % | 1.8498 | 41.40 % | 1.386e-03 | 76.1 s |
| 7 | 1.7793 | 40.93 % | 1.8096 | 42.40 % | 1.005e-03 | 99.3 s |
| 8 | 1.7424 | 42.51 % | 1.7899 | 43.60 % | 6.242e-04 | 101.8 s |
| 9 | 1.7032 | 43.89 % | 1.7491 | 45.00 % | 3.014e-04 | 116.6 s |
| 10 | 1.6784 | 45.18 % | 1.7467 | **45.20 %** | 8.574e-05 | 161.7 s |

**观察**：

1. 训练损失与验证损失**单调下降**，未出现过拟合（验证损失始终跟随训练损失下降）；
2. 验证准确率在最后一轮**仍在上升**，说明模型**远未收敛**——这是 10 轮 + 4,500 张样本的直接后果，
   也正是需要在 I1 扩大规模的依据；
3. 学习率按 warmup + 余弦调度**精确衰减**（1e-3 → 2e-3 平台 → 8.57e-5），
   与 `figures/training_curves.png`（图 1）子图 (c) 完全对应。

---

## 6. 三类结果与文件的对应关系

| 要求的分类 | 目录 | 关键文件 | 索引/说明文档 |
|---|---|---|---|
| **图表** | `figures/` | 图 1–图 4 共 4 张 PNG | `figures/figures_index.md` |
| **日志** | `logs/` | `train.log`（12 个执行阶段）、`evaluate.log`、`history.json` | `logs/README.md` |
| **模型** | `models/` | `best.pt`、`last.pt`（完整检查点） | `models/README.md` |
| 指标（补充） | `metrics/` | `config.json`、`metrics.csv`、`test_metrics.json`、`per_class_metrics.json`、`eval_metrics.json` | `metrics/README.md` |

---

## 7. 复现命令

### 7.1 复现本次 MVP 实验（CPU 约 19 分钟）

```bash
cd 03_code
python train.py \
  --run-tag mvp_atto_subset5k \
  --model atto --epochs 10 --batch-size 128 --lr 2e-3 --warmup-epochs 2 --no-mixup \
  --train-subset-size 5000 --test-subset-size 2000 \
  --data-root ../04_data/cifar10 --output-dir ../05_results \
  --exp-name mvp_atto_subset5k --plot-every 2 --num-workers 0 --seed 42

# 独立评估（自动从检查点恢复模型配置）
python evaluate.py --ckpt ../05_results/mvp_atto_subset5k/models/best.pt
```

> `--run-tag mvp_atto_subset5k` 保证产物写入 `05_results/mvp_atto_subset5k/`，
> 与其它实验**完全隔离**；重复运行只会覆盖本实验自身的文件。

### 7.2 全量实验（推荐在 GPU / Colab 上运行）

```bash
cd 03_code
python train.py --run-tag full_convnext_tiny_cifar10 \
  --model tiny --epochs 60 --batch-size 128 \
  --data-root ../04_data/cifar10 --output-dir ../05_results
```

在 Colab 中也可直接使用 `03_code/notebooks/ConvNeXt_CIFAR10_Colab.ipynb` 一键运行
（该 Notebook 会先生成与工程**逐字一致**的源码，再自动完成全部流程）。

---

## 8. 与其它部分的引用关系

| 关联 | 说明 |
|---|---|
| 数据 | 本实验读取 `../../04_data/cifar10`；划分索引见 `../../04_data/splits/split_info.json`（`mvp_subset_*` 三个文件即本实验所用索引） |
| 代码 | 全部产物由 `../../03_code` 生成；实验参数见 `metrics/config.json` |
| 模型 → 指标 | `metrics/test_metrics.json` 由 `models/best.pt` 在测试集上评估得到 |
| 模型 → 图表 | 图 2、图 3 由 `models/best.pt` 的预测生成；图 1 的数据来自 `logs/history.json` |
| 日志 → 时间 | `logs/train.log` 的时间戳给出"何时得到该指标"的证据 |
| 上游小结 | `../../02_ai_chats/I0_summary.md` |
