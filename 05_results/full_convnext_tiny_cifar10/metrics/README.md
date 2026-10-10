# metrics —— 指标文件字段说明（`full_convnext_tiny_cifar10`）

> **归属任务**：T6（实验运行与结果收集）→【指标】产物；其中 `metrics.csv` 为约定的"一行式汇总表"。
> 本目录 5 个文件全部由 `03_code` 自动写出，无手工编辑。

## 1. 文件清单与用途

| 文件 | 生成者 | 用途 |
|---|---|---|
| `config.json` | `train.py` 启动时 | 本次运行**全部超参数**快照（可复现性证据），同时被写入检查点 |
| `metrics.csv` | `train.py` 结束时 | **一行式**指标汇总表（含逐类准确率列），多次实验可直接纵向拼接成对比/消融表 |
| `test_metrics.json` | `train.py` 结束时 | 训练流程内的最终测试指标（**无 TTA**） |
| `per_class_metrics.json` | `train.py` / `evaluate.py` | 逐类准确率 + 10×10 混淆矩阵（**最近一次评估的口径**：本实验为 TTA） |
| `eval_metrics.json` | `evaluate.py --tta` | 独立评估结果（**TTA**：原图 + 水平翻转平均 softmax） |

## 2. `config.json` —— 超参数快照

| 分组 | 关键字段 |
|---|---|
| `data` | `dataset`(CIFAR10)、`data_root`、`num_classes`、`val_ratio`(0.1)、`train_subset_size`/`test_subset_size`(null = 全量)、`random_erase`(0.25)、`rand_augment`(false)、`mean`/`std` |
| `model` | `name`(tiny)、`depths`/`dims`(null = 用预设)、`drop_path_rate`(0.1)、`layer_scale_init_value`(1e-6)、`stem_mode`(conv)、`head_init_scale`(1.0) |
| `train` | `epochs`(60)、`batch_size`(128)、`lr`(1e-3)、`min_lr`(1e-5)、`weight_decay`(0.05)、`warmup_epochs`(5)、`label_smoothing`(0.1)、`grad_clip`(1.0)、`use_mixup`(true)、`mixup_prob`(0.5)、`mixup_alpha`(0.8)、`cutmix_alpha`(1.0)、`amp`(true)、**`amp_fallback`(true)**、`seed`(42)、`num_workers`(0)、`plot_every` |
| 顶层 | `output_dir`(`../05_results`)、`device`(auto→cuda)、`exp_name`、`run_tag` |

> `amp_fallback`：AMP 安全网。fp16 的 GradScaler scale 崩塌（< 1）时自动降级为 fp32 续训，避免"假收敛"（见 `03_code/README.md` 第 12.1 节）。本实验未触发（用的是 bf16）。

## 3. `metrics.csv` —— 一行式汇总表（本实验实际值）

| 列（节选） | 值 | 列（节选） | 值 |
|---|---|---|---|
| `experiment` | convnext_cifar10 | `is_full_scale` | **True** |
| `model_preset` / `num_params` | tiny / 27825802 | `split` | train 45000 / val 5000 / test 10000 |
| `device` | CUDA:0 \| RTX 4060 Laptop \| 显存 8.0 GB | `batch_size_actual` / `effective_batch_size` | 128 / 128 |
| `epochs_completed` | 60 | `best_val_acc` / `best_val_epoch` | 93.5 / 59 |
| `test_top1_acc` | **92.66** | `test_top5_acc` / `test_loss` | 99.56 / 0.684162 |
| `macro_accuracy` | 92.67 | `total_training_time` | 1h 55m 08s |
| `amp_dtype` | **bfloat16** | `amp_min_scale` | 2097152.0 |
| `acc_airplane … acc_truck` | 93.5 / 98.2 / 90.2 / 82.1 / 92.9 / 87.9 / 95.9 / 94.6 / 96.3 / 95.1 | `seed` | 42 |

> 列顺序由 `03_code/src/utils/csv_logger.py` 显式固定，因此**可以把多个实验的 `metrics.csv` 纵向拼接**成报告中的对比表。

## 4. `test_metrics.json` —— 训练流程内的最终测试指标（无 TTA）

| 字段 | 值/说明 |
|---|---|
| `architecture` | preset=tiny、stem_mode=conv、drop_path_rate=0.1、num_params=27,825,802 |
| `dataset` | name=CIFAR10、is_synthetic=false、split、**is_full_scale=true**、data_root |
| `device` / `gpu_total_memory_gb` | CUDA:0 RTX 4060 Laptop / 8.0 |
| `batch_size_requested` / `actual` / `effective_batch_size` | 128 / 128 / 128 |
| `epochs` / `epochs_completed` | 60 / **60** |
| `best_val_acc` / `best_val_epoch` | **93.5** / 59 |
| `test_metrics` | `loss`=0.684162、`top1_acc`=**92.66**、`top5_acc`=99.56、`num_samples`=10000 |
| `evaluation` | `tta`=false、`tta_views`=["identity"]、说明 loss 不计 TTA |
| `per_class_accuracy` / `macro_accuracy` | 各类准确率 / 92.67 |
| `total_training_time` | 1h 55m 08s（**续训进程**的墙钟时间；含首次中断段合计约 3h05m，见 `../logs/README.md` 第 3 节） |
| `amp` | `{enabled: true, dtype: "bfloat16", min_scale: 2097152.0, fell_back_to_fp32: false}` |

## 5. `per_class_metrics.json` 与 `eval_metrics.json`

两者结构相近（后者额外含 `checkpoint`、`batch_size`、`model` 字段）：

| 字段 | 说明 |
|---|---|
| `num_samples` / `num_correct` / `accuracy` | 10,000 / 9,287 / **92.87**（TTA 口径） |
| `macro_accuracy` | 各类准确率的算术平均（92.87） |
| `per_class_accuracy` | 10 个类各自的准确率（0~1 小数） |
| `confusion_matrix` | 10×10 整数计数，行 = 真实类、列 = 预测类，每行合计 1,000 |
| `class_names` | 类别顺序：`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck` |
| `evaluation.tta` / `tta_views` | true / ["identity","hflip"]（`test_metrics.json` 中为 false，二者口径不同，勿混用） |

> **口径提示**：`test_metrics.json` = 无 TTA（92.66 %）；`per_class_metrics.json` / `eval_metrics.json` = TTA（92.87 %）。
> 图 2、图 3 与后者同源。报告引用时请注明是否使用 TTA。
