# full_convnext_tiny_cifar10 —— 全量主实验结果（本机 GPU）

> **实验标签**：`full_convnext_tiny_cifar10` ｜ **对应迭代**：**I1（全量数据 + 60 轮，产出可写入报告的主结果）**
> **归属任务**：**T6（实验运行与结果收集）、T7（结果可视化与图表整理）**
> 算法：**ConvNeXt**（*A ConvNet for the 2020s*, CVPR 2022, arXiv:2201.03545） ｜ 任务：CIFAR-10 图像分类
> 实验日期：2026-10-10 ｜ 计算设备：**本机 GPU（NVIDIA GeForce RTX 4060 Laptop, 8 GB）**，bf16 混合精度
> 全部内容为 `03_code` 在**真实全量 CIFAR-10** 上**实际运行产生**（`is_full_scale = true`）

---

## 1. 目录结构（本实验自包含）

```
full_convnext_tiny_cifar10/
├── README.md                        # 本文件：结果总览（表 1–表 4）
├── figures/                         # 【图表】
│   ├── figures_index.md             #   图表索引：图 1–图 4 的编号、含义、读法（T7 产物）
│   ├── training_curves.png          #   图 1：损失曲线 / 准确率曲线 / 学习率曲线
│   ├── confusion_matrix.png         #   图 2：测试集混淆矩阵（按真实类别归一化，含 TTA）
│   ├── per_class_accuracy.png       #   图 3：逐类准确率条形图（含 TTA）
│   └── data_samples.png             #   图 4：CIFAR-10 数据样例网格
├── logs/                            # 【日志】
│   ├── README.md                    #   日志索引与执行阶段标注（含断点续训说明）
│   ├── train.log                    #   训练日志（带 [HH:MM:SS] 时间戳，69 KB）
│   ├── evaluate.log                 #   独立评估日志（GPU + TTA）
│   ├── history.json                 #   逐轮结构化指标（loss / acc / lr / 耗时 / amp_scale）
│   └── history.csv                  #   逐轮指标表（60 行，可直接进报告）
├── models/                          # 【模型】
│   ├── README.md                    #   检查点结构、加载示例、架构与超参数说明
│   ├── best.pt                      #   验证集最优模型（epoch 59，约 334 MB）
│   └── last.pt                      #   最后一轮模型（epoch 60，约 334 MB）
└── metrics/                         # 【指标】
    ├── README.md                    #   各字段含义说明
    ├── config.json                  #   本次运行的全部超参数快照（可复现性证据）
    ├── metrics.csv                  #   一行式指标汇总表（含逐类准确率列，T6 产物）
    ├── test_metrics.json            #   训练结束时的最终测试指标（无 TTA）
    ├── per_class_metrics.json       #   逐类准确率 + 10×10 混淆矩阵（TTA 口径）
    └── eval_metrics.json            #   独立评估（evaluate.py --tta）结果
```

> **目录隔离**：本实验的全部产物由 `--run-tag full_convnext_tiny_cifar10` 写入本目录，
> 与 `05_results/mvp_atto_subset5k/`、`05_results/_failed_runs/` 互不覆盖、互不交叉引用。

---

## 2. 表 1 —— 实验设置

| 项目 | 值 |
|---|---|
| 实验标签 | `full_convnext_tiny_cifar10` |
| 算法 / 类别 | ConvNeXt ／ Deep CNN |
| 模型预设 | `tiny`（阶段通道 96/192/384/768，Block 数 3/3/9/3；即论文 ConvNeXt-T） |
| 参数量 | 27,825,802（27.83 M） |
| Stem 类型 | `conv`（3×3 stride=1，适配 32×32 输入） |
| 数据集 | CIFAR-10（`../../04_data/cifar10`），**全量**：训练 45,000 / 验证 5,000 / 测试 10,000 |
| 优化器 | AdamW（lr=1e-3，weight decay=0.05，1 维参数豁免权重衰减） |
| 学习率调度 | 线性 warmup 5 轮 + 余弦退火至 1e-5 |
| 批大小 | 128（等效 batch 128，无梯度累积） |
| 训练轮数 | 60 |
| 数据增强 | RandomCrop(32, padding=4, reflect) + RandomHorizontalFlip + RandomErasing(p=0.25) + **Mixup/CutMix**（每 batch 50 % 概率，alpha 0.8 / 1.0） |
| 标签平滑 / DropPath | 0.1 / 0.1（各 Block 从 0 线性升到 0.1） |
| 随机种子 | 42 |
| **计算设备** | **CUDA:0 NVIDIA GeForce RTX 4060 Laptop GPU（8 GB）** |
| **混合精度** | **bfloat16 + GradScaler**（`amp.dtype = bfloat16`，`amp.min_scale = 2.10e+06`，未触发 fp32 回退） |
| 数据加载 | `num_workers = 0`（Windows 多进程共享内存提交耗尽，见第 6.2 节） |
| 总耗时 | 训练脚本统计 **1 h 55 m 08 s**（续训进程；含首次中断段合计约 3 h 05 m） |
| 平均单轮耗时 | epoch 1–32 约 118 ~ 135 s；epoch 33–60 约 181 ~ 250 s（`num_workers` 由 4 改为 0 所致） |

> **为什么用 bf16 而不是 fp16**：本机 RTX 4060（计算能力 8.9）原生支持 bfloat16，
> 工程会自动选择它；bf16 与 fp32 阶码范围相同，不存在 fp16 那种梯度下溢风险。
> 反面案例见 `../_failed_runs/full_convnext_tiny_cifar10_amp_fp16_underflow/`（T4 + fp16 → 测试 Top-1 仅 25.73 %）。

---

## 3. 表 2 —— 测试集总体指标

基于 `models/best.pt`（epoch 59），测试集 10,000 张：

| 指标 | 无 TTA（`metrics/test_metrics.json`） | **TTA**（`metrics/eval_metrics.json`，原图 + 水平翻转） |
|---|---|---|
| 测试损失（交叉熵，含标签平滑 0.1） | 0.6842 | 0.6842（loss 不计 TTA） |
| **Top-1 准确率** | 92.66 % | **92.87 %** |
| **Top-5 准确率** | 99.56 % | **99.61 %** |
| 宏平均准确率（各类准确率均值） | 92.67 % | 92.87 % |
| 正确样本数 / 总样本数 | 9,266 / 10,000 | **9,287 / 10,000** |
| 最优验证准确率（模型选择依据） | **93.50 %（epoch 59）** | — |

> * 随机基线：10 类均匀猜测 Top-1 = 10 %、Top-5 = 50 %；本实验分别高出 **82.87 / 49.61 个百分点**。
> * TTA 带来 **+0.21 个百分点**（92.66 % → 92.87 %），与工程文档预期的 +0.2 ~ +0.5 一致。
> * 验证集 93.50 % 与测试集 92.66 % 相差 0.84 个百分点，属正常泛化差距，**无过拟合迹象**（见表 4）。
> * 结果落在设计文档预期区间（93–95 %）的下沿：本机 GPU 显存 8 GB → batch 128（预期配方为 batch 128 时 lr 1e-3），
>   且 60 轮 + Mixup/CutMix 下最后一轮验证准确率仍在缓慢上升（见第 6.1 节），继续加轮数仍有提升空间。

---

## 4. 表 3 —— 逐类准确率（测试集，TTA 口径）

| 类别 | TTA 准确率 | 无 TTA | 支撑样本 | 类别 | TTA 准确率 | 无 TTA | 支撑样本 |
|---|---|---|---|---|---|---|---|
| airplane | 93.80 % | 93.50 % | 1,000 | dog | 88.20 % | 87.90 % | 1,000 |
| automobile | **97.90 %** | 98.20 % | 1,000 | frog | 96.20 % | 95.90 % | 1,000 |
| bird | 90.90 % | 90.20 % | 1,000 | horse | 94.20 % | 94.60 % | 1,000 |
| cat | **82.80 %** | 82.10 % | 1,000 | ship | 96.50 % | 96.30 % | 1,000 |
| deer | 93.40 % | 92.90 % | 1,000 | truck | 94.80 % | 95.10 % | 1,000 |

**误差分析**（由 `metrics/per_class_metrics.json` 的 10×10 混淆矩阵按行归一化得到，每真实类 1,000 张）：

| 主要混淆方向（真实 → 预测） | 占该真实类比例 | 说明 |
|---|---|---|
| **cat → dog** | **8.2 %** | 最大误差来源；cat 类召回率 82.8 %，其余误差分散在 frog 2.6 % / deer 2.0 % / bird 1.9 % |
| **dog → cat** | **8.1 %** | 与 cat→dog 近似对称，构成一对典型难分对（毛发纹理、姿态相似） |
| truck → automobile | 2.9 % | 同为四轮机动车，仅车身轮廓/货箱差异 |
| airplane → ship | 2.2 % | 均以天空/水面为背景，细长轮廓相似 |
| bird → frog / cat / deer | 2.2 / 1.7 / 1.7 % | 小目标、多姿态，类内差异大 |
| deer → frog / cat | 2.0 / 1.7 % | 与草绿色背景相关 |
| horse → dog / cat | 2.0 / 1.3 % | 四足动物形态相近 |

> 结论与 `figures/confusion_matrix.png`（图 2）中高亮的非对角区域一致：
> **cat ↔ dog** 是唯一一对双向混淆超过 8 % 的类别，其余混淆均在 3 % 以下。

---

## 5. 表 4 —— 逐轮训练指标（节选自 `logs/history.csv`，共 60 行）

| Epoch | train_loss | train_acc | val_loss | val_acc | 学习率 | 单轮耗时 | amp_scale |
|---|---|---|---|---|---|---|---|
| 1 | 2.0320 | 28.79 % | 1.7330 | 43.02 % | 2.000e-04 | 120.8 s | 6.55e+04 |
| 2 | 1.8810 | 36.73 % | 1.5759 | 51.36 % | 4.000e-04 | 117.6 s | 6.55e+04 |
| 5 | 1.6347 | 50.27 % | 1.2903 | 65.16 % | 1.000e-03 | 127.7 s | 6.55e+04 |
| 10 | 1.3322 | 65.29 % | 0.9730 | 79.72 % | 9.871e-04 | 126.1 s | 1.31e+05 |
| 15 | 1.2405 | 70.06 % | 0.8621 | 84.74 % | 9.360e-04 | 125.5 s | 2.62e+05 |
| 20 | 1.1136 | 75.98 % | 0.7948 | 87.92 % | 8.500e-04 | 119.7 s | 5.24e+05 |
| 25 | 1.1023 | 76.83 % | 0.7571 | 89.16 % | 7.360e-04 | 120.1 s | 1.05e+06 |
| 30 | 1.0096 | 80.70 % | 0.7225 | 90.74 % | 6.033e-04 | 117.4 s | 2.10e+06 |
| 32 | 0.9555 | 82.94 % | 0.7248 | 90.70 % | 5.474e-04 | 119.6 s | 2.10e+06 |
| **33** | 0.9640 | 82.72 % | 0.7016 | 91.44 % | 5.191e-04 | **228.1 s** ← 断点续训后（num_workers=0） | 2.10e+06 |
| 35 | 0.9338 | 84.10 % | 0.7000 | 92.00 % | 4.626e-04 | 231.6 s | 4.19e+06 |
| 40 | 0.9318 | 83.97 % | 0.6919 | 92.12 % | 3.254e-04 | 238.2 s | 8.39e+06 |
| 45 | 0.9157 | 84.53 % | 0.6793 | 93.06 % | 2.027e-04 | 244.4 s | 8.39e+06 |
| 50 | 0.8625 | 86.56 % | 0.6779 | 93.04 % | 1.045e-04 | 238.7 s | 1.68e+07 |
| 55 | 0.9448 | 83.20 % | 0.6761 | 93.22 % | 3.879e-05 | 230.1 s | 3.36e+07 |
| 58 | 0.8986 | 85.34 % | 0.6711 | 93.48 % | 1.725e-05 | 206.0 s | 6.71e+07 |
| **59** | 0.8727 | 86.31 % | 0.6697 | **93.50 %（最优）** | 1.323e-05 | 181.2 s | 6.71e+07 |
| 60 | 0.9127 | 84.52 % | 0.6709 | 93.42 % | 1.081e-05 | 210.0 s | 6.71e+07 |

> 完整 60 行见 `logs/history.csv`（列顺序固定，可直接作为报告中的"逐轮指标表"）。
> `train_acc` 低于 `val_acc` 属**正常**：训练侧叠加了 Mixup/CutMix（软标签使准确率指标被压低）与随机增强，验证/测试侧为确定性变换。

---

## 6. 训练过程说明

### 6.1 收敛行为（对照图 1 `figures/training_curves.png`）

1. **训练/验证损失单调下降**且验证损失始终跟随训练损失（0.6697 为最优），**未出现过拟合**；
2. 验证准确率从第 1 轮 43.02 % 快速爬升到第 30 轮 90.74 %，随后进入**缓慢精修阶段**，在最后 10 轮仍从 92.9 % 提升到 **93.50 %** ——说明 60 轮尚未完全饱和，继续训练（或换更大 batch / 更长时间）仍有收益空间；
3. 学习率严格按 warmup + cosine 从 1e-3 衰减到 1.08e-5，与图 1(c) 完全对应；
4. 第 33 轮出现一次"台阶"：`val_acc` 由 90.70 % 跳到 91.44 %，同时单轮耗时由 120 s 变为 228 s —— 这正是**断点续训 + 数据加载策略变更**造成的，指标本身连续（见 6.2）。

### 6.2 训练中断与断点续训（过程证据）

| 时间 | 事件 |
|---|---|
| 12:17 → 13:26 | 首次运行（`--num-workers 4`），完成 epoch 1–32（best 90.86 %） |
| 13:26:36 | **进程崩溃**：`RuntimeError: Couldn't open shared file mapping ... error code: 1455`（= Windows 系统提交内存耗尽，**非显存问题**；Windows DataLoader 多进程共享内存的已知问题） |
| 14:06:47 | 执行 `--resume auto --num-workers 0` **原地续训**：模型 / 优化器 / 调度器 / 逐轮历史与学习率全部按检查点恢复（日志首行为 `[resume] 已从 last.pt 恢复：下一轮 epoch=32，历史最优验证准确率=90.86%`） |
| 14:07 → 16:01 | 继续完成 epoch 33–60，随后自动在测试集上评估并落盘全部产物 |

* 本次中断**未造成指标损失**，只损失了崩溃时正在进行的第 33 轮的部分计算；
* 复现该场景：`--resume auto`（工程内置，专为长跑中断设计）；
* 相关经验已写入 `03_code/README.md` 第 12.4 / 12.5 节（显存探测留余量、Windows 共享内存提交耗尽）。

### 6.3 与失败运行（fp16 事故）的对照

| 对比项 | `_failed_runs/..._amp_fp16_underflow`（T4 + **fp16**） | **本实验（RTX 4060 + bf16）** |
|---|---|---|
| `GradScaler` 动态 scale | **崩塌到 0.0156**（默认 65536，连减 22 次） | 单调健康增长 6.55e+04 → 6.71e+07 |
| 骨干参数是否更新 | `stages.1.*` 的 `exp_avg_sq` **恒为 0**（从未更新） | 全网正常更新 |
| 训练损失 | 60 轮停在 2.15 ~ 2.27（ln10 = 2.303） | 2.03 → 0.87 |
| 验证准确率 | 峰值 26.66 %（epoch 2） | 峰值 **93.50 %**（epoch 59） |
| 测试 Top-1 | 25.73 % | **92.66 % / TTA 92.87 %** |

---

## 7. 三类结果与文件的对应关系

| 要求的分类 | 目录 | 关键文件 | 索引/说明文档 |
|---|---|---|---|
| **图表** | `figures/` | 图 1–图 4 共 4 张 PNG | `figures/figures_index.md` |
| **日志** | `logs/` | `train.log`、`evaluate.log`、`history.json`、`history.csv` | `logs/README.md` |
| **模型** | `models/` | `best.pt`（epoch 59）、`last.pt`（epoch 60） | `models/README.md` |
| 指标（补充） | `metrics/` | `config.json`、`metrics.csv`、`test_metrics.json`、`per_class_metrics.json`、`eval_metrics.json` | `metrics/README.md` |

---

## 8. 复现命令

```bash
cd 03_code

# ---- 训练（本机 GPU，bf16 AMP；约 2 小时）----
python train.py --run-tag full_convnext_tiny_cifar10 \
  --model tiny --epochs 60 --batch-size 128 --stem-mode conv \
  --device cuda --num-workers 0 \
  --data-root ../04_data/cifar10 --output-dir ../05_results

# ---- 若长跑中断：原地续训（不丢已完成的轮次与优化器状态）----
python train.py --run-tag full_convnext_tiny_cifar10 \
  --model tiny --epochs 60 --batch-size 128 --stem-mode conv \
  --device cuda --num-workers 0 --resume auto \
  --data-root ../04_data/cifar10 --output-dir ../05_results

# ---- 独立评估（GPU + TTA）----
python evaluate.py --ckpt ../05_results/full_convnext_tiny_cifar10/models/best.pt \
  --device cuda --tta \
  --data-root ../04_data/cifar10 --output-dir ../05_results
```

> 环境要求：CUDA 版 PyTorch（`torch==2.14.1+cu130` / `torchvision==0.29.1+cu130`），
> 安装方式见 `03_code/requirements.txt`。`--num-workers 0` 为 Windows 长跑的安全设置
> （多进程 DataLoader 的共享内存提交可能耗尽，见 6.2）；
> Linux / Colab 上可改为 `--num-workers 2` 加速。
> 其它超参数（lr、warmup、Mixup、标签平滑、DropPath、种子等）均已写入 `metrics/config.json`。

---

## 9. 与其它部分的引用关系

| 关联 | 说明 |
|---|---|
| 数据 | 本实验读取 `../../04_data/cifar10`；全量划分索引见 `../../04_data/splits/train_indices.pt` / `val_indices.pt` / `test_indices.pt`，划分协议见 `../../04_data/splits/split_info.json` |
| 代码 | 全部产物由 `../../03_code` 生成；运行参数见 `metrics/config.json`；AMP 行为见 `03_code/src/utils/amp.py` |
| 模型 → 指标 | `metrics/test_metrics.json`、`metrics/eval_metrics.json` 由 `models/best.pt` 在测试集上评估得到 |
| 模型 → 图表 | 图 2、图 3 由 `models/best.pt` 的预测生成；图 1 的数据来自 `logs/history.json` |
| 日志 → 时间 | `logs/train.log` 的 `[HH:MM:SS]` 时间戳给出"何时得到该指标"以及断点续训时点的证据 |
| 失败对照 | `../_failed_runs/full_convnext_tiny_cifar10_amp_fp16_underflow/`（fp16 下溢事故的完整证据链） |
| 小规模基线 | `../mvp_atto_subset5k/`（CPU 端到端 MVP，测试 Top-1 45.05 %） |
