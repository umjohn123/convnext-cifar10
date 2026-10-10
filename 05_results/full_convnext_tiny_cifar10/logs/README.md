# logs —— 日志索引与执行阶段标注（`full_convnext_tiny_cifar10`）

> **归属任务**：T6（实验运行与结果收集）
> 本目录保存本次全量主实验的**过程日志**：何时开始、每轮指标、何时评估、是否有异常，
> 全部带 `[HH:MM:SS]` 时间戳，可直接作为"可追溯性证据"。

## 1. 文件清单

| 文件 | 大小 | 内容 |
|---|---|---|
| `train.log` | 69 KB | 训练全过程日志：环境与设备、显存探测、数据划分、模型摘要、逐轮 train/val 指标与 `amp_scale`、检查点保存、最终测试评估与总结 |
| `evaluate.log` | 6 KB | `evaluate.py --tta` 独立评估日志（GPU + TTA：原图 + 水平翻转） |
| `history.json` | 11 KB | 逐轮结构化指标（`epoch / train_loss / train_acc / val_loss / val_acc / lr / epoch_time / amp_scale`） |
| `history.csv` | 4 KB | 同上，CSV 形式（60 行，列顺序固定），**可直接作为报告中的"逐轮指标表"** |
| `crash_err1455_stderr.log` | 3 KB | **过程证据**：首次运行（`--num-workers 4`）在第 33 轮崩溃时的原始 Python traceback，末行为 `RuntimeError: Couldn't open shared file mapping: <torch_21920_...>, error code: <1455>`（= Windows 系统提交内存耗尽，非显存问题） |

## 2. 执行阶段（`train.log` 的关键节点）

| 时间 | 阶段 |
|---|---|
| 14:06:47 | 启动；`[resume] 已从 last.pt 恢复：下一轮 epoch=32，历史最优验证准确率=90.86%`（本次为**断点续训**，详见第 3 节） |
| 14:06:47 | 显存探测：`最终确定 batch size = 128（探针峰值显存 2883 MB，当前可用显存 7000 MB，GPU 总显存 8.0 GB）` |
| 14:06:47 | `[device] CUDA:0 | NVIDIA GeForce RTX 4060 Laptop GPU | 显存 8.0 GB` |
| 14:06:47 | 训练配置：`AMP=True（autocast bfloat16 + GradScaler） | Mixup/CutMix=True | 参数量=27.83M` |
| 14:07 → 16:01 | epoch 33 → 60 逐轮训练与验证（每轮写 `history.json` / `history.csv` 与 `last.pt`，刷新最优时写 `best.pt`） |
| 16:01:36 | 训练结束；载入 `best.pt`（epoch 59）→ 测试集评估 → 图 2/图 3 → `test_metrics.json` / `metrics.csv` |
| 16:01:36 | `全部完成！总耗时 1h 55m 08s`、`最终测试集 Top-1 准确率：92.66%` |
| 16:04 → 16:05 | 独立的 TTA 评估（`evaluate.log`）：`Top-1 = 92.87% | Top-5 = 99.61%`，并刷新图 2/图 3 与 `eval_metrics.json` |

## 3. 重要说明：断点续训与单轮耗时台阶

* 本次训练**被中断过一次**：首次运行（12:17 → 13:26，`--num-workers 4`）完成 epoch 1–32 后，
  于 13:26:36 因 Windows DataLoader 多进程共享内存提交耗尽而崩溃
  （`RuntimeError: Couldn't open shared file mapping ... error code: 1455`，**非显存问题**）。
* 14:06:47 用 `--resume auto --num-workers 0` **原地续训**，从 epoch 33 继续；
  模型、优化器、学习率调度器与逐轮历史均按检查点恢复（故 `history.csv` 的 60 行是**连续且完整**的）。
* 因此日志与 `history.csv` 中会出现一处"台阶"：**epoch ≤ 32 的单轮耗时约 118 ~ 135 s**，
  **epoch ≥ 33 约 181 ~ 250 s**（`num_workers` 由 4 变为 0 导致数据加载搬到主进程）。
  该变化**不影响指标**，只影响耗时；模型精度曲线在 epoch 32 → 33 处连续（90.70 % → 91.44 %）。
* `train.log` 采用**追加写**（`mode="a"`），所以文件里同时保留了崩溃前那一段（12:17–13:26）的日志，
  便于对照排查；
* 崩溃的**原始 traceback** 已保留在 `crash_err1455_stderr.log`（3 KB），可直接作为"中断原因"的一手证据；
  同一现象与处置也记录在 `03_code/README.md` 第 12.5 节。

## 4. 关键证据字段

| 想确认… | 查看 |
|---|---|
| 是否真的在用 GPU | `train.log` 的 `[device] CUDA:0 NVIDIA GeForce RTX 4060 ...` |
| 混合精度是否健康 | 每轮日志末尾的 `amp_scale`（本实验 6.55e+04 → 6.71e+07，单调增长；崩塌到 <1 即为异常，见 `03_code/README.md` 第 12.1 节） |
| 是否发生过 fp32 回退 | `train.log` 是否有 `[amp] ... 自动降级为 fp32` 告警（本实验**没有**） |
| 何时得到某指标 | 对应时间戳（如 `[Epoch 59/60] 完成 | ... val_top1=93.50%`） |
| 批次/等效批次 | `开始训练：epochs=60 ... batch_size=128 x accum=1 (等效 batch=128)` |
| 数据规模是否为全量 | `[data] 数据集 划分完成 | 训练 45000 张 | 验证 5000 张 | 测试 10000 张` |
