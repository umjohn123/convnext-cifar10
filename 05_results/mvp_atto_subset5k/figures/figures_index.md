# figures_index —— 图表索引

| 项目 | 内容 |
|---|---|
| 文档名称 | 图表索引（本实验全部可视化结果） |
| **归属任务** | **T7 —— 结果可视化与图表整理** |
| 所属实验 | `mvp_atto_subset5k` |
| 归档位置 | `05_results/mvp_atto_subset5k/figures/figures_index.md` |
| 生成方式 | 全部由 `03_code/src/utils/visualize.py` 在训练/评估过程中自动生成（PNG，200 DPI），**非手工绘制** |

---

## 1. 图表清单（编号可直接用于报告引用）

| 编号 | 文件名 | 图表内容 | 生成时机 | 生成函数 |
|:---:|---|---|---|---|
| **图 1** | `training_curves.png` | 三联图：**(a)** 训练/验证损失曲线　**(b)** 训练/验证准确率曲线（自动标注最优验证准确率点）　**(c)** 学习率调度曲线（warmup + cosine，对数纵轴） | 每 `--plot-every` 轮刷新一次，训练结束时绘制最终版 | `plot_training_curves()` |
| **图 2** | `confusion_matrix.png` | 测试集混淆矩阵，**按真实类别（行）归一化**，每格标注比例，颜色深浅表示占比 | 训练结束后的最终测试评估；`evaluate.py` 独立评估时刷新（二者结果一致） | `plot_confusion_matrix(normalize=True)` |
| **图 3** | `per_class_accuracy.png` | 测试集逐类准确率条形图，柱顶标注数值，标题含宏平均准确率 | 同上 | `plot_per_class_accuracy()` |
| **图 4** | `data_samples.png` | CIFAR-10 数据样例网格（16 张，2×8），已反归一化还原真实颜色并标注类别名 | 训练开始前（用于确认数据与增强是否正确） | `plot_data_samples()` |

**编号规则**：图 1–图 3 依"报告论述顺序"编号（先训练过程 → 再整体评估 → 再分类别分析），
图 4 为数据说明性插图。报告正文引用时请直接使用"图 n"。

---

## 2. 各图表的读法与说明

### 2.1 图 1 —— 训练曲线（`training_curves.png`）

* **横轴**：训练轮次 `epoch`（从 1 起）。
* **(a) Loss**：训练损失按每个 batch 的样本数加权平均；验证损失为整轮平均。
  * 正常收敛形态：两条曲线同步下降，末期趋于平缓；
  * `train` 明显低于 `val` 且 `val` 开始上升 → 过拟合信号。
* **(b) Accuracy**：训练准确率与验证准确率（百分数）。黑点标注验证集最优轮次。
  * `train` 与 `val` 差距过大 → 数据量不足或正则化不足；
  * 曲线剧烈震荡 → 学习率过大或 batch size 过小。
* **(c) LR schedule**：学习率随轮次的变化（对数纵轴）。前 `--warmup-epochs` 轮线性上升，
  之后按余弦从 `--lr` 衰减到 `--min-lr`，用于确认调度器按预期工作。
* **本实验对应数值**：见实验 `README.md` 的表 4（逐轮指标）。
* **原始数据**：`../logs/history.json`（逐轮 loss / acc / lr / 耗时）。

### 2.2 图 2 —— 混淆矩阵（`confusion_matrix.png`）

* **纵轴**：真实类别；**横轴**：预测类别。类别顺序固定为
  `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`。
* 已**按行归一化**，因此对角线数值 = 该类别的**召回率（逐类准确率）**。
* 常见混淆对（视觉上相似的类别）：`cat ↔ dog`、`deer ↔ horse`、`automobile ↔ truck`。
* **用途**：定位最易混淆的类别对，为报告第 4 项的误差分析提供依据。
* **原始数据**：`../metrics/per_class_metrics.json` 的 `confusion_matrix` 字段（10×10 整数计数）。

### 2.3 图 3 —— 逐类准确率（`per_class_accuracy.png`）

* 直接给出每一类的准确率以及**宏平均**（各类准确率的算术平均）。
* 与图 2 的对角线一致，但更便于横向比较各类难度。
* **本实验结论**：最好 ship 61.50 %，最差 bird 24.34 %（详见实验 `README.md` 表 3）。

### 2.4 图 4 —— 数据样例（`data_samples.png`）

* 用于验证：数据是否正确下载、归一化参数是否正确、增强后图像是否正常。
* 若该图颜色明显失真，说明 `mean/std` 与训练时不一致。
* **数据来源**：`../../../04_data/cifar10` 的验证集（不做增强的确定性变换）。

---

## 3. 图表与其它文件的对应关系

| 关联 | 说明 |
|---|---|
| 图 ← 模型 | 图 2、图 3 使用 `../models/best.pt`（验证集最优模型）在测试集上的预测 |
| 图 ← 日志 | 图 1 的原始数值来自 `../logs/history.json` |
| 图 ← 指标 | 图中的准确率数值与 `../metrics/test_metrics.json`、`../metrics/per_class_metrics.json` 完全一致 |
| 图 ← 数据 | 图 4 取自 `../../../04_data/cifar10` |
| 图 → 报告 | 四张图可直接嵌入英文报告第 4 项「Experiment settings and results」 |

---

## 4. 复现说明

图表不是手工制作的产物，而是训练流程的自动输出。要重新生成：

```bash
cd 03_code
python train.py --run-tag mvp_atto_subset5k --model atto --epochs 10 ... \
                --data-root ../04_data/cifar10 --output-dir ../05_results
python evaluate.py --ckpt ../05_results/mvp_atto_subset5k/models/best.pt
```

> 只要随机种子、数据划分与超参数一致，重新生成的四张图与当前版本一致。
