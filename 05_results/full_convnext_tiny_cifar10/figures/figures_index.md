# figures_index —— 图表索引（全量主实验）

| 项目 | 内容 |
|---|---|
| 文档名称 | 图表索引（本实验全部可视化结果） |
| **归属任务** | **T7 —— 结果可视化与图表整理** |
| 所属实验 | `full_convnext_tiny_cifar10` |
| 归档位置 | `05_results/full_convnext_tiny_cifar10/figures/figures_index.md` |
| 生成方式 | 全部由 `03_code/src/utils/visualize.py` 在训练/评估过程中自动生成（PNG，200 DPI），**非手工绘制** |
| 数据规模 | 全量 CIFAR-10（训练 45,000 / 验证 5,000 / 测试 10,000） |

---

## 1. 图表清单（编号可直接用于报告引用）

| 编号 | 文件名 | 图表内容 | 生成时机 | 生成函数 |
|:---:|---|---|---|---|
| **图 1** | `training_curves.png` | 三联图：**(a)** 训练/验证损失曲线　**(b)** 训练/验证准确率曲线（自动标注最优验证准确率点）　**(c)** 学习率调度曲线（warmup + cosine，对数纵轴） | 每 `--plot-every`（默认 10）轮刷新，训练结束时绘制最终版 | `plot_training_curves()` |
| **图 2** | `confusion_matrix.png` | 测试集混淆矩阵，**按真实类别（行）归一化**，每格标注比例，颜色深浅表示占比（TTA 口径，与 `metrics/per_class_metrics.json` 同源） | 训练结束后的最终测试评估；`evaluate.py --tta` 独立评估时刷新 | `plot_confusion_matrix(normalize=True)` |
| **图 3** | `per_class_accuracy.png` | 测试集逐类准确率条形图，柱顶标注数值，标题含宏平均准确率（92.87 %） | 同上 | `plot_per_class_accuracy()` |
| **图 4** | `data_samples.png` | CIFAR-10 数据样例网格（16 张，2×8），已反归一化还原真实颜色并标注类别名 | 训练开始前（用于确认数据与增强是否正确） | `plot_data_samples()` |

**编号规则**：图 1–图 3 依"报告论述顺序"编号（先训练过程 → 再整体评估 → 再分类别分析），
图 4 为数据说明性插图。报告正文引用时请直接使用"图 n"。

---

## 2. 各图表的读法与说明

### 2.1 图 1 —— 训练曲线（`training_curves.png`）

* **横轴**：训练轮次 `epoch`（1 ~ 60）。
* **(a) Loss**：训练损失按 batch 样本数加权平均（含 Mixup 软标签，故数值高于验证损失）；验证损失为整轮平均。
  * 本实验形态：两条曲线同步下降，验证损失从 1.7330 单调降到 **0.6697（epoch 59）**，**无过拟合**。
* **(b) Accuracy**：训练/验证准确率（百分数），黑点标注验证集最优轮次。
  * 本实验形态：验证准确率 43.02 %（epoch 1）→ 90.74 %（epoch 30）→ **93.50 %（epoch 59，黑点）**；
    训练准确率低于验证准确率属正常（训练侧有 Mixup/CutMix 与随机增强）。
* **(c) LR schedule**：学习率随轮次变化（对数纵轴）。前 5 轮线性升到 1e-3，之后余弦衰减到 1.08e-5，
  与 `logs/history.csv` 的 `lr` 列一致。
* **原始数据**：`../logs/history.json`（逐轮 loss / acc / lr / 耗时 / amp_scale）。

### 2.2 图 2 —— 混淆矩阵（`confusion_matrix.png`）

* **纵轴**：真实类别；**横轴**：预测类别；顺序固定为
  `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`。
* 已**按行归一化**（每真实类 1,000 张），对角线数值 = 该类召回率。
* 本实验最显著的非对角区域：**cat → dog 8.2 %** 与 **dog → cat 8.1 %**（唯有一对双向混淆超过 8 %）；
  其次为 truck → automobile 2.9 %、airplane → ship 2.2 %、bird → frog 2.2 %。
* **原始数据**：`../metrics/per_class_metrics.json` 的 `confusion_matrix` 字段（10×10 整数计数）。

### 2.3 图 3 —— 逐类准确率（`per_class_accuracy.png`）

* 直接给出每一类准确率与**宏平均**（92.87 %）。
* 最好：automobile 97.90 %、ship 96.50 %、frog 96.20 %；最差：**cat 82.80 %**、dog 88.20 %、bird 90.90 %。
* 与图 2 的对角线一致，详见实验 `README.md` 表 3。

### 2.4 图 4 —— 数据样例（`data_samples.png`）

* 用于验证：数据是否正确下载与解压、归一化参数是否正确、增强后图像是否正常。
* 若颜色明显失真，说明 `mean/std` 与训练时不一致（本工程为 CIFAR-10 标准值 0.4914/0.4822/0.4465 与 0.2470/0.2435/0.2616）。
* **数据来源**：`../../../04_data/cifar10` 的验证集（不做增强的确定性变换）。

---

## 3. 图表与其它文件的对应关系

| 关联 | 说明 |
|---|---|
| 图 ← 模型 | 图 2、图 3 使用 `../models/best.pt`（epoch 59，验证集最优）在测试集上的预测（含 TTA） |
| 图 ← 日志 | 图 1 的原始数值来自 `../logs/history.json` |
| 图 ← 指标 | 图中数值与 `../metrics/eval_metrics.json`、`../metrics/per_class_metrics.json` 完全一致（Top-1 92.87 %） |
| 图 ← 数据 | 图 4 取自 `../../../04_data/cifar10` |
| 图 → 报告 | 四张图可直接嵌入报告第 4 项「Experiment settings and results」 |

---

## 4. 复现说明

```bash
cd 03_code
python train.py --run-tag full_convnext_tiny_cifar10 --model tiny --epochs 60 --batch-size 128 \
                --stem-mode conv --device cuda --num-workers 0 \
                --data-root ../04_data/cifar10 --output-dir ../05_results
python evaluate.py --ckpt ../05_results/full_convnext_tiny_cifar10/models/best.pt --device cuda --tta \
                --data-root ../04_data/cifar10 --output-dir ../05_results
```

> 只要随机种子（42）、数据划分与超参数一致，重新生成的四张图与当前版本一致。
