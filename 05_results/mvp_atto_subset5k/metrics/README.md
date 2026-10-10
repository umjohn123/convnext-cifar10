# metrics —— 指标

> 结果分类：**【指标】**（对「图表 / 日志 / 模型」三类的补充）
> 以 JSON 形式保存**机器可读**的实验设置与定量结果，供报告引用与结果核对。

---

## 1. 指标文件清单

| 文件名 | 内容 | 产生阶段 |
|---|---|---|
| `config.json` | 本次运行的**全部超参数**（数据 / 模型 / 训练三段配置的完整快照），可复现性证据 | 训练开始前（`train.py`） |
| `metrics.csv` | **一行式指标汇总表**（40 列）：30 列实验设置与总体指标 + 10 列逐类准确率 `acc_<类别>`。列顺序固定，把多个实验的该文件纵向拼接即可直接得到报告中的对比/消融表格 | 训练结束时（`train.py`，由 `src/utils/csv_logger.py::write_metrics_csv()` 生成） |
| `test_metrics.json` | **最终测试结果汇总**：实验标识、算法与架构信息、数据集规模、设备、batch size 决策、训练轮数、最优验证准确率、测试集 loss / Top-1 / Top-5、逐类准确率、宏平均、总耗时、随机种子 | 训练结束时（`train.py`） |
| `per_class_metrics.json` | 逐类详细指标：样本数、正确数、总体准确率、宏平均、**逐类准确率字典**、**混淆矩阵（10×10 计数）**、类别名称顺序 | 训练结束时（`train.py`） |
| `eval_metrics.json` | 由 `evaluate.py` **独立**评估测试集得到的指标（检查点路径、模型信息、Top-1/Top-5、逐类准确率、混淆矩阵） | 评估阶段（`evaluate.py`） |

---

## 2. 关键字段说明

### 2.1 `test_metrics.json`

| 字段 | 含义 |
|---|---|
| `experiment` | 实验名称（与 `--exp-name` 一致），用于区分不同配置的运行 |
| `algorithm` | 算法名称与文献出处（ConvNeXt, CVPR 2022, arXiv:2201.03545） |
| `architecture` | 模型预设、Stem 类型、DropPath、各阶段通道/深度、参数量 |
| `dataset` | 数据集名称、**实际划分规模**、`is_full_scale`（是否为全量数据）、子集参数、数据根目录 |
| `device` / `gpu_total_memory_gb` | 计算设备描述与显存总量（CPU 时为 0） |
| `batch_size_requested` / `batch_size_actual` | 期望批大小与实际探测结果 |
| `grad_accum_steps` / `effective_batch_size` | 梯度累积步数与等效批大小 |
| `epochs` / `epochs_completed` | 计划轮数与实际完成轮数 |
| `best_val_acc` / `best_val_epoch` | 验证集最优准确率与对应轮次（模型选择依据） |
| `test_metrics.top1_acc` / `top5_acc` | **测试集 Top-1 / Top-5 准确率（百分数）** |
| `per_class_accuracy` / `macro_accuracy` | 逐类准确率与宏平均 |
| `total_training_time` / `seed` | 总耗时与全局随机种子 |

### 2.2 `per_class_metrics.json`

| 字段 | 含义 |
|---|---|
| `confusion_matrix` | 10×10 整数矩阵，`cm[i][j]` = 真实类别 i 被预测为 j 的样本数（行=真实，列=预测） |
| `per_class_accuracy` | 各类准确率（= 混淆矩阵行归一化后的对角线） |
| `macro_accuracy` | 各类准确率的算术平均 |
| `class_names` | 类别顺序，**列顺序与混淆矩阵/准确率列表严格对应** |

> ⚠️ 引用混淆矩阵时务必核对 `class_names` 顺序，避免类别错位。

### 2.3 重要提示：`is_full_scale`

`test_metrics.json` 中的 `dataset.is_full_scale` 用于标记本次实验是否为**全量数据**实验：

* `true` —— 全量 CIFAR-10（训练 45000 / 验证 5000 / 测试 10000），结果可直接用于报告结论；
* `false` —— 使用了 `--train-subset-size` / `--test-subset-size` 的**快速验证子集**，
  仅用于验证流程正确性，**其准确率不能与全量结果或文献结果直接比较**。

---

## 3. 与其它部分的关系

| 关联 | 说明 |
|---|---|
| 指标 ← 数据 | `config.json` 记录 `data_root` 与实际规模；`../../../04_data/splits/split_info.json` 给出对应的索引文件 |
| 指标 ← 日志 | `test_metrics.json` 中的数值与 `../logs/train.log` 的 `[test]` 行一致（日志是时间证据） |
| 指标 ← 模型 | 全部测试指标由 `../models/best.pt` 在测试集上推理得到 |
| 指标 → 图表 | `../figures/confusion_matrix.png` 由 `per_class_metrics.json` 的混淆矩阵绘制 |
| 指标 → 报告 | 直接对应报告第 4 项的"定量结果"表格 |
