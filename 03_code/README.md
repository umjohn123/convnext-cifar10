# ConvNeXt on CIFAR-10 — PyTorch 完整工程

> 课程：**CISC3024 Pattern Recognition** ｜ 作业：**AIassign2（AI Assignment #1）**
> 目标算法：**ConvNeXt**（*A ConvNet for the 2020s*, **CVPR 2022**, arXiv:2201.03545）
> 算法类别：**Deep CNN** ｜ 应用领域：**计算机视觉 / 模式识别（图像分类）**

本工程全部代码由 AI 生成，实现 ConvNeXt 在 CIFAR-10 上的**训练、验证、测试评估与可视化**完整流水线，
可直接在 **Google Colab** 中运行（也可在本地 CPU/CUDA 环境运行）。

---

## 0. 与作业目录的对应关系（代码 / 数据 / 结果）

本工程是作业目录中的**代码部分**，与数据、结果两部分通过**命令行参数**显式关联，引用关系如下：

```
Assignment2/
├── 03_code/      ← 【代码】本工程：模型定义、训练、评估、可视化（全部由 AI 生成）
│      └─ 通过 --data-root  指向 04_data   （读取数据集）
│      └─ 通过 --output-dir 指向 05_results（写出模型 / 日志 / 图表 / 指标）
│      └─ 通过 --run-tag    隔离实验       （决定 05_results 下的实验子目录名）
├── 04_data/      ← 【数据】CIFAR-10 原始数据集 + 划分索引 + 数据说明（T5）
└── 05_results/   ← 【结果】每个实验一个独立子目录，内部再分
                       figures(图表) / logs(日志) / models(模型) / metrics(指标)（T6/T7）
```

| 引用方向 | 参数 | 默认值 | 说明 |
|---|---|---|---|
| 代码 → 数据 | `--data-root` | `./data`（Colab 友好） | 数据集读取位置；作业目录结构下建议用 `../04_data/cifar10` |
| 代码 → 结果 | `--output-dir` | `./outputs` | 产物根目录；作业目录结构下建议用 `../05_results` |
| 结果 → 实验 | `--run-tag` | `None` | **实验隔离**：设置后全部产物写入 `<output-dir>/<run-tag>/`，多次实验互不覆盖、互不交叉引用 |

> 采用「参数显式引用」而非硬编码相对路径，是为了让同一份代码既能在 Colab 里独立运行，
> 又能无缝接入 `03_code / 04_data / 05_results` 的作业目录结构。

---

## 1. 目录结构

```
03_code/
├── README.md                     # 本说明文档
├── requirements.txt              # Python 依赖清单（Colab 已内置，无需安装）
├── configs.py                    # 全局配置：数据 / 模型 / 训练超参数 + 命令行解析
├── train.py                      # ★ 训练入口：训练 → 验证 → 测试评估 → 出图
├── evaluate.py                   # ★ 独立评估入口：用已保存检查点评估测试集
├── src/                          # 源码包
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cifar10.py            # CIFAR-10 常量、变换与数据底座（每个 split 只构造一次）
│   │   ├── synthetic.py          # 合成数据：**零下载**，用于网络受限时的冒烟测试
│   │   └── loader.py             # 划分协议、TransformSubset 与 DataLoader 组装
│   ├── models/
│   │   ├── __init__.py
│   │   └── convnext.py           # ConvNeXt 模型定义（论文关键模块全部保留）
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py            # 训练循环：AdamW + warmup/cosine + AMP + 梯度累积 + 断点续训
│   │   └── evaluator.py          # 评估：Top-1/Top-5、逐类准确率、混淆矩阵
│   └── utils/
│       ├── __init__.py
│       ├── amp.py                # 混合精度（AMP）跨版本兼容层
│       ├── csv_logger.py         # CSV 导出：history.csv（逐轮）与 metrics.csv（结果汇总）
│       ├── device.py             # 设备选择 + **显存自适应 batch size** 探测
│       ├── logger.py             # 控制台 + 文件双输出日志
│       ├── misc.py               # AverageMeter / JSON 读写 / 参数量统计
│       ├── mixup.py              # Mixup 与 CutMix 数据增强
│       ├── seed.py               # 随机种子固定（可复现性）
│       ├── tta.py                # 测试时增强（原图 + 水平翻转，平均 softmax 概率）
│       └── visualize.py          # 训练曲线、混淆矩阵、逐类准确率、数据样例
├── scripts/
│   └── fetch_cifar10.py          # CIFAR-10 高速下载器（镜像测速 + 多连接分片 + MD5 校验 + 解压）
├── notebooks/
│   ├── ConvNeXt_CIFAR10_Colab.ipynb   # ★ 开箱即用的 Colab Notebook（自包含）
│   └── build_colab_notebook.py        # 由本工程源码自动生成上述 Notebook 的脚本（含完整性自检）
└── outputs/                      # 运行时自动生成的全部产物（按"结果三分类"组织）
    ├── figures/                  # 【图表】曲线、混淆矩阵、逐类准确率、数据样例
    ├── logs/                     # 【日志】train.log / evaluate.log / history.json
    ├── models/                   # 【模型】best.pt / last.pt
    └── metrics/                  # 【指标】config.json / metrics.csv / test_metrics.json / per_class_metrics.json
```

> 输出目录名与作业 `05_results` 的结构**完全一致**：
> `figures`=图表、`logs`=日志、`models`=模型、`metrics`=指标。
> 再配合 `--run-tag`，产物会准确落入 `05_results/<实验标签>/`，无需任何后处理。

---

## 2. 快速开始

### 2.1 Google Colab（推荐，一键运行）

打开 `notebooks/ConvNeXt_CIFAR10_Colab.ipynb`（可用 Colab 直接打开），
依次执行单元格即可：Notebook 会**自动创建整个工程、下载 CIFAR-10、探测显存、开始训练并展示曲线**。

或者手动在 Colab 中执行：

```python
!pip install -q torchvision          # Colab 通常已内置，可跳过
!python train.py --epochs 60 --model tiny --batch-size 128 --no-amp
```

> 建议在 Colab 中开启 GPU：`代码执行程序 → 更改运行时类型 → T4 GPU`。
>
> **修改源码后必须重新生成 Notebook**：Notebook 由源码自动生成，若两者不同步，
> Colab 上会出现 `ModuleNotFoundError`（例如新增了 `src/utils/tta.py` 但 Notebook 里没有该文件）。
> 重新生成并自检：
>
> ```bash
> cd 03_code
> python notebooks/build_colab_notebook.py     # 生成后会自动做"完整性 + 逐字一致性"自检
> ```
>
> 生成器会**自动发现**工程中所有 `.py` 文件（不再依赖手工清单），因此新增模块不会再被遗漏；
> 自检不通过时会以非零退出码报错并列出具体问题。

### 2.2 本地 / 服务器

```bash
pip install -r requirements.txt        # 默认 PyPI 上 Windows 的 torch 是 CPU 版

# ---- 需要 GPU 时：安装 CUDA 版 PyTorch（Windows / Python 3.14 实测可用） ---- #
pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu130 \
            "torch==2.14.1+cu130" "torchvision==0.29.1+cu130"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 独立运行（数据与产物都落在本工程目录下，适合 Colab / 单机实验）
#   --device cuda 显式指定用 GPU（默认 auto 也会自动选到 CUDA）
python train.py --epochs 60 --model tiny --batch-size 128 --device cuda
python evaluate.py --ckpt ./outputs/models/best.pt --device cuda

# 按作业目录结构运行（代码 03_code / 数据 04_data / 结果 05_results），并用 --run-tag 隔离实验
python train.py --run-tag full_convnext_tiny_cifar10 \
                --model tiny --epochs 60 --batch-size 128 --device cuda \
                --data-root ../04_data/cifar10 \
                --output-dir ../05_results
python evaluate.py --ckpt ../05_results/full_convnext_tiny_cifar10/models/best.pt \
                --device cuda --tta \
                --data-root ../04_data/cifar10 \
                --output-dir ../05_results
```

> **本地有 GPU 时不要加 `--no-amp`**（`--no-amp` 只是 T4 等**只能跑 fp16** 的设备上的兜底做法）：
> 工程会自动选择安全的精度 —— Ampere 及以后（如 RTX 30/40 系、L4、A100）用 **bfloat16**，
> 日志中会打印 `AMP=True（autocast bfloat16 + GradScaler）`。
> **实测（RTX 4060 Laptop 8 GB，完整全量实验）**：`--model tiny --batch-size 128 --device cuda`，
> 60 轮全量 CIFAR-10（45,000 张）共耗约 3 小时（其中一次因 Windows 共享内存问题中断后续训 1h55m），
> 单轮耗时：`--num-workers 4` 约 2 分钟、`--num-workers 0` 约 4 分钟；
> 最优验证 **93.50 %（epoch 59）**、测试集 **Top-1 92.66 %（TTA 92.87 %）**、Top-5 99.56 %，
> `amp_scale` 全程健康（6.55e+04 → 6.71e+07）。产物见 `../05_results/full_convnext_tiny_cifar10/`。

### 2.3 快速冒烟测试（零下载，约 30 秒 ~ 1 分钟）

```bash
# 推荐：完全不下载任何数据，用本地合成数据 + 受限 step 数验证"环境 + 代码链路"
python train.py --run-tag smoke_test --dataset synthetic \
                --synthetic-train-size 512 --synthetic-test-size 256 \
                --model atto --batch-size 32 --max-steps-per-epoch 4 \
                --epochs 2 --exp-name smoke_test
```

> **为什么冒烟测试默认不用真实数据集**：CIFAR-10 归档有 162 MB，官方源单连接在部分网络下
> 只有约 0.1 MB/s（下载要 25 分钟），仅这一步就会让冒烟测试远超预期时长。
> 合成数据由本地生成（`src/data/synthetic.py`），形态与 CIFAR-10 完全一致（32×32×3、10 类、
> 返回 `(PIL.Image, int)`），且类间带有可学习的结构信号 —— 因此它验证的**是同一条真实代码路径**
> （同一套变换、模型、训练循环、评估与绘图），只是不需要下载。
>
> **为什么还要 `--max-steps-per-epoch`**：移除下载后，瓶颈转移到计算。
> Colab 纯 CPU 运行时只有 2 vCPU，即使很小的数据集，完整跑完一个 epoch 也可能超时。
> 该参数给训练量设了与数据集规模、硬件性能无关的上界（N 个 step）。
>
> 实测（本机 20 核 CPU）：`--max-steps-per-epoch 4` + 512/256 合成数据，端到端 **19.8 秒**。
>
> ⚠️ 合成数据的指标**没有科学意义**，禁止写入报告结果；结果文件中通过
> `dataset.name = "synthetic"` 与 `is_full_scale = false` 双重标记。

### 2.4 CPU 上的小规模真实实验

```bash
# 快速验证子集：5000 张训练池 + 2000 张测试集
python train.py --run-tag mvp_atto_subset5k \
                --model atto --epochs 10 --batch-size 128 \
                --train-subset-size 5000 --test-subset-size 2000 \
                --data-root ../04_data/cifar10 --output-dir ../05_results \
                --exp-name mvp_atto_subset5k
```

> ⚠️ 使用 `--train-subset-size` / `--test-subset-size` 时，程序会在日志中打印**显式警告**，
> 并把子集规模写入 `<实验目录>/metrics/test_metrics.json`（`is_full_scale: false`），
> 以避免小规模结果被误当作全量结果引用。
>
> 上述命令的产物已归档于 `05_results/mvp_atto_subset5k/`（真实运行结果，测试 Top-1 = 45.05 %）。

---

## 3. 命令行参数（常用）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | `tiny` | ConvNeXt 规模：`atto` / `femto` / `pico` / `nano` / `tiny` |
| `--epochs` | `60` | 训练轮数 |
| `--batch-size` | `128` | **期望**的 batch size（显存不足会自动下调） |
| `--no-auto-batch-size` | – | 关闭显存自适应探测 |
| `--lr` | `1e-3` | AdamW 基础学习率 |
| `--weight-decay` | `0.05` | 权重衰减（1 维参数自动豁免） |
| `--warmup-epochs` | `5` | 线性热身轮数（其后余弦退火） |
| `--label-smoothing` | `0.1` | 标签平滑系数 |
| `--stem-mode` | `conv` | `conv`=3x3 s1（适配 32x32）｜`patchify`=论文原版 4x4 s4 |
| `--drop-path` | `0.1` | Stochastic Depth 丢弃率 |
| `--seed` | `42` | 全局随机种子 |
| `--no-amp` | – | 关闭混合精度训练（**T4 上建议开启**：fp16 会导致本模型浅层梯度下溢，见第 12 节） |
| `--no-amp-fallback` | – | 禁用 AMP 安全网：GradScaler 的 scale 崩塌时不再自动降级为 fp32 |
| `--no-mixup` | – | 关闭 Mixup / CutMix |
| `--tta` | – | 最终测试评估启用 **TTA**（原图 + 水平翻转，平均 softmax 概率） |
| `--no-tta-hflip` | – | TTA 只保留原图视角（需与 `--tta` 一起看，等价于关闭翻转分支） |
| `--data-root` | `./data` | 数据集目录（作业结构下建议 `../04_data/cifar10`） |
| `--dataset` | `CIFAR10` | `CIFAR10`（真实，需下载）或 `synthetic`（本地合成，**零下载**，仅用于冒烟测试） |
| `--synthetic-train-size` | `4096` | 合成数据训练集规模（`--dataset synthetic` 时生效） |
| `--synthetic-test-size` | `2048` | 合成数据测试集规模（`--dataset synthetic` 时生效） |
| `--max-steps-per-epoch` | `None` | 每个 epoch 最多训练多少个 step（**限时冒烟测试用**，默认不限制） |
| `--train-subset-size` | `None` | 快速验证子集：从训练池抽取的样本数（None = 全量 50000） |
| `--test-subset-size` | `None` | 快速验证子集：从测试集抽取的样本数（None = 全量 10000） |
| `--plot-every` | `10` | 每多少轮刷新一次曲线图（0 = 仅结束时绘制） |
| `--early-stop-patience` | `0` | 早停耐心值（0 = 不启用） |
| `--exp-name` | `convnext_cifar10` | 实验名称（写入日志与指标文件） |
| `--run-tag` | `None` | **实验隔离标签**：产物写入 `<output-dir>/<run-tag>/`（正式实验建议始终指定） |
| `--resume` | – | 断点续训的 checkpoint 路径；填 `auto` 则自动从本次实验的 `last.pt` 续训 |
| `--output-dir` | `./outputs` | 产物输出根目录（作业结构下建议 `../05_results`） |

---

## 4. 算法实现说明（ConvNeXt 关键模块）

代码完整保留论文《A ConvNet for the 2020s》的核心设计：

| 编号 | 论文关键设计 | 本工程实现位置 |
|---|---|---|
| 1 | 阶段堆叠比例 (3, 3, 9, 3)、通道数逐阶段翻倍 | `ConvNeXt.__init__`（`depths` / `dims`） |
| 2 | 卷积式 Stem（patchify） | `ConvNeXt.__init__`（`stem_mode`） |
| 3 | 倒置瓶颈 Inverted Bottleneck（升维 4 倍 → 深度卷积 → 降维） | `ConvNeXtBlock` |
| 4 | 7x7 深度卷积（大卷积核） | `ConvNeXtBlock.dwconv` |
| 5 | GELU 替代 ReLU、每块仅 1 个激活 | `ConvNeXtBlock.act` |
| 6 | LayerNorm 替代 BatchNorm | `LayerNorm` |
| 7 | 独立下采样层（LayerNorm + 2x2 卷积） | `ConvNeXtDownsample` |
| 8 | Layer Scale（初值 1e-6） | `ConvNeXtBlock.gamma` |
| 9 | Stochastic Depth | `DropPath` |
| 10 | 更小的分类头初始化缩放 | `ConvNeXt.head_init_scale` |

**针对 CIFAR-10 的适配说明**：ImageNet 输入为 224x224，论文 Stem 为 4x4、stride=4；
CIFAR-10 仅 32x32，若沿用 stride=4 则特征图会迅速缩到 8→4→2→1，空间信息损失过大。
因此默认使用 **3x3、stride=1 的卷积 Stem**，四个阶段分辨率为 **32 → 16 → 8 → 4**，
在保留论文全部关键模块的前提下适配小分辨率输入。用 `--stem-mode patchify` 可切回论文原版。

**模型规模与参数量（实测）**：

| 预设 | depths | dims | 参数量 |
|---|---|---|---|
| atto | (2,2,6,2) | (40,80,160,320) | ≈ 3.7 M |
| femto | (2,2,6,2) | (48,96,192,384) | ≈ 5.2 M |
| pico | (2,2,6,2) | (64,128,256,512) | ≈ 9.0 M |
| nano | (2,2,8,2) | (80,160,320,640) | ≈ 15.6 M |
| **tiny**（对应论文 ConvNeXt-T） | **(3,3,9,3)** | **(96,192,384,768)** | **≈ 27.8 M** |

---

## 5. 数据与预处理

* **数据集**：CIFAR-10（32x32 RGB，10 类，每类 6000 张）。
* **划分**：官方 50000 张训练图按固定种子（`--seed`）划分为
  **训练 45000 / 验证 5000**；官方测试集 **10000 张**完全不参与训练与调参。
* **训练增强**：`RandomCrop(32, padding=4, reflect)` + `RandomHorizontalFlip`
  + `RandomErasing(p=0.25)` + `ToTensor` + `Normalize`，另可选 `RandAugment`。
* **评估变换**：`ToTensor` + `Normalize`（确定性，保证可复现）。
* **归一化参数**：mean=(0.4914, 0.4822, 0.4465)，std=(0.2470, 0.2435, 0.2616)。
* 数据首次运行会自动下载到 `./data`（可用 `--data-root` 修改）。

### 5.1 数据集获取与下载加速

CIFAR-10 归档为 162 MB，官方源 **单连接**在部分网络环境下速率很低
（本机实测 0.11 MB/s → 需约 25 分钟）。该源**支持 HTTP Range**，
因此可用多连接分片把聚合带宽提升一个量级。加速器已产品化：

```bash
python scripts/fetch_cifar10.py --dest ./data                 # 下载到 ./data（默认 8 连接 × 4 MiB 分块）
python scripts/fetch_cifar10.py --dest ./data --probe-only     # 只体检镜像、不下载
python scripts/fetch_cifar10.py --dest ../04_data/cifar10      # 作业目录结构
python scripts/fetch_cifar10.py --threads 16 --chunk-mb 2      # 带宽充裕时进一步加大并发
```

脚本行为：**镜像体检（长度 + 速率 + Range）** → **多连接分块下载（断点续传 + 指数退避重试）**
→ **大小 + MD5 双重校验** → **自动解压**；任一环节失败都会**自动回退**（降并发 → 单连接 → 换镜像）。

**为什么要先校验远端长度**：镜像被截断时，多连接会以"分片下载不完整"这种误导性错误收场。
实测 `cs231n.stanford.edu` 上那份归档只有 `170497704` 字节（比官方少 **367 字节**），
于是 `Range: bytes=170497704-170498070` 被服务器判为越界并返回 **416**。
脚本用 `HEAD` 读取服务器**自己的** `Content-Length` 后可直接判定该镜像损坏并剔除，
避免用 16 个连接跑完 162 MB 才发现 MD5 不对。

另外两点实现细节：镜像探测采用**带截止时间的有界读取**（若镜像忽略 Range 头返回 200，
无界 `read()` 会读完整个归档；而只靠 `urlopen(timeout=...)` 也不够 —— 慢速服务器每次都能在
超时内返回少量数据，累加起来仍可能耗时数十秒，实测 42 s）；所有数据请求都带
`Accept-Encoding: identity`，避免压缩传输打乱 Range 偏移语义。

### 5.2 数据加载性能优化（实测）

`torchvision.datasets.CIFAR10` 每次构造都会对全部 6 个批次文件（约 186 MB）做 MD5 完整性校验，
并把 pickle 反序列化到内存。原实现为"带增强训练集 / 不带增强验证集 / 测试集"各构造一个实例，
导致 CIFAR-10 被实例化 **3 次**、完整性校验触发 **6 次**、训练集 pickle（146 MB）被读入内存 **2 份**。

现改为「**每个 split 只构造一个 `transform=None` 的底座** + `TransformSubset` 叠加不同变换」，
**划分与索引语义完全不变**（已用 `04_data/splits/*.pt` 做逐项回归校验），
热路径耗时由 **4.5 s 降至 1.16 s（约 3.9×）**。

---

## 6. 显存自适应 batch size（重要）

Colab 的 GPU 型号不固定（T4 15GB / L4 24GB / A100 40GB），写死 batch size 极易 OOM。
本工程在训练前会做一次**真实的 前向 + 反向 + 优化器更新 探针**（`src/utils/device.py`）：

1. 从 `--batch-size` 期望值开始探测；若显存不足则**逐次减半**，直到跑通或触及 `min_batch_size`；
2. 若开启 `auto`，成功后再**逐级倍增**，但要求 `探针峰值 ≤ 当前可用显存 × 70 %`
   （`_PROBE_FREE_RATIO`）—— 必须为真实训练留余量，否则 Windows/WDDM 的显存过度提交
   会让探针"通过"一个物理上放不下的档位（详见第 12.4 节）；
3. 若 batch size 被下调，则自动启用**梯度累积**，使等效 batch size 仍接近期望值；
4. 全过程写入日志（含探针峰值显存、当前可用显存与 GPU 总显存），作为可复现性证据。

---

## 7. 训练与评估增强特性

### 7.1 测试时增强（TTA）

`--tta` 启用「**原图 + 水平翻转**」两路测试时增强：两个视角分别前向后，
对 **softmax 概率**取算术平均，再用平均概率计算 Top-1 / Top-5。

| 项目 | 说明 |
|---|---|
| 实现位置 | `src/utils/tta.py`；被 `src/engine/evaluator.py` 的 `evaluate()` / `collect_predictions()` 调用 |
| 与训练增强的区别 | 训练时 `RandomHorizontalFlip` 是**随机采样**；TTA 的翻转是**成对、确定性**的，不引入随机性，结果完全可复现 |
| 代价与收益 | 推理耗时约 ×2；CIFAR-10 上通常可带来 **+0.2 ~ +0.5 个百分点**的 Top-1 |
| 口径一致性 | `loss` 始终由**不做 TTA 的原始前向**计算（与训练期验证、非 TTA 评估严格可比）；仅 Top-k 准确率使用 TTA 平均概率。结果文件中的 `evaluation.tta` 字段标记本次是否启用 |
| 注意事项 | `evaluate()` 与 `collect_predictions()` 的 `tta` 必须取相同值，否则混淆矩阵/逐类准确率会与总体 Top-1 对不上（代码中已由同一变量控制） |

### 7.2 断点续训与自动恢复

* `--resume <路径>`：从指定检查点恢复模型、优化器、调度器、AMP scaler 与训练历史；
* `--resume auto`：若**本次实验目录**下已存在 `models/last.pt`，则自动续训，否则从头开始。
  该选项专为 Colab 免费实例中途断连的场景设计 —— 重连后执行**同一条命令**即可接着训练。

### 7.3 逐轮增量落盘（防丢失）

训练历史在**每一轮结束时**同时写入 `logs/history.json` 与 `logs/history.csv`（而非等训练全部结束）。
这样即使会话中断，已完成的逐轮指标也不会丢失；`history.csv` 可直接作为报告中的"逐轮指标表"。

### 7.4 混合精度（AMP）：精度选择与 fp16 下溢防护

**精度选择**（`src/utils/amp.py`）：Ampere 及以后（计算能力 ≥ 8.0，如 L4 / A100）**自动使用 bfloat16**；
T4（7.5）不支持原生 bf16，因此回退 fp16。日志中会显式打印实际精度，例如
`AMP=True（autocast bfloat16 + GradScaler）`。

**为什么默认不用 fp16**：本模型 LayerScale 初值为 `1e-6`，残差块内参数的梯度都带 γ 因子，
18 个块叠加后浅层 Block 的真实梯度只有 `1e-6` 甚至更低（实测 `1.8e-06`）。
fp16 的最小正规数约 `6.1e-05`、最小次正规数约 `6.0e-08`；一旦 GradScaler 的动态 scale 因
fp16 溢出被连续减半压到 `0.0156` 量级，梯度乘上 scale 后仍低于 fp16 可表示范围，
会被直接抹成 **精确的 0** —— 这些参数从此再也收不到梯度。

**症状**（曾经真实发生过，详见 `05_results/_failed_runs/`）：`train_loss` 60 轮停在 2.2 附近、
`val_acc` 峰值只有 26.7 %、测试 Top-1 = 25.7 %，但日志**没有任何报错**。
事后从检查点可读出：`scaler_state.scale = 0.0156`（默认 65536），
且 `stages.1.*.pwconv1.weight` 的 `exp_avg_sq` **恒为 0**（一次都没更新过）。

**三重防护**：

1. 每轮在日志与 `history.csv` 的 `amp_scale` 列输出 GradScaler 的动态 scale（正常应接近 65536）；
2. `amp_scale` 跌破 `AMP_SCALE_FLOOR = 1.0` 时打印告警，并**自动关闭 AMP、以 fp32 继续训练**
   （`trainer._track_amp_scale()`，可用 `--no-amp-fallback` 关闭）；
3. 检查点与 `metrics/test_metrics.json` 中保存 `amp{enabled, dtype, min_scale, fell_back_to_fp32}`，
   `metrics.csv` 增加 `amp_dtype` / `amp_min_scale` 两列 —— 事后可据此判定一次运行的指标是否可信。

**结论性建议**：Colab T4 上跑正式实验请加 `--no-amp`（Notebook 3-C 单元格已默认加上）；
换到 L4 / A100 时可直接去掉（自动用 bf16）。

---

## 8. 输出产物

| 分类 | 文件 | 内容 |
|---|---|---|
| 【模型】 | `outputs/models/best.pt` | 验证集准确率最优的模型（含优化器/调度器/历史/配置） |
| 【模型】 | `outputs/models/last.pt` | 最后一轮模型，可用 `--resume` 断点续训 |
| 【日志】 | `outputs/logs/train.log` | 完整训练日志（带时间戳） |
| 【日志】 | `outputs/logs/evaluate.log` | 独立评估日志 |
| 【日志】 | `outputs/logs/history.json` | 逐轮的 train/val 损失、准确率、学习率、耗时 |
| 【日志】 | `outputs/logs/history.csv` | 同上，CSV 形式（准确率已换算为百分数，另含 `amp_scale` 列），可直接作为报告的逐轮指标表 |
| 【图表】 | `outputs/figures/training_curves.png` | **(a) 损失曲线 (b) 准确率曲线 (c) 学习率曲线** |
| 【图表】 | `outputs/figures/confusion_matrix.png` | 测试集归一化混淆矩阵 |
| 【图表】 | `outputs/figures/per_class_accuracy.png` | 测试集逐类准确率 |
| 【图表】 | `outputs/figures/data_samples.png` | CIFAR-10 数据样例网格 |
| 【指标】 | `outputs/metrics/test_metrics.json` | 最终测试指标（Top-1/Top-5、逐类准确率、宏平均等）+ `amp` 字段（精度/最小 scale/是否降级） |
| 【指标】 | `outputs/metrics/per_class_metrics.json` | 逐类准确率与混淆矩阵 |
| 【指标】 | `outputs/metrics/metrics.csv` | 一行式指标汇总表（含逐类准确率列，可拼接成对比表） |
| 【指标】 | `outputs/metrics/config.json` | 本次运行的全部超参数（可复现性证据） |

---

## 9. 预期结果

在 CIFAR-10 上使用默认配方（`--model tiny --epochs 60`，AdamW + warmup/cosine
+ label smoothing + Mixup/CutMix + RandomErasing，seed=42），
测试集 **Top-1 准确率通常在 93%–95%** 区间（取决于 GPU 与 batch size）。
若只需短时间验证，`--model pico --epochs 30` 也可在较短时间内达到约 90% 以上。

> ⚠️ **如果结果只在 25 % 左右**：不要怀疑数据或模型，先看是否是 **fp16 AMP 的梯度下溢**
> （第 7.4 节）。判据有三条——① 日志中 `AMP=True（autocast float16 ...）`；
> ② 每轮 `amp_scale` 远小于 65536（例如 `0.0156`）；③ `train_loss` 60 轮停在 2.2 附近不下降。
> T4 上请加 `--no-amp` 重跑；L4 / A100 上程序会自动使用 bf16，不会出现该问题。
> 一次真实的失败案例已完整归档在 `05_results/_failed_runs/full_convnext_tiny_cifar10_amp_fp16_underflow/`。

---

## 10. 可复现性

* 固定 Python / NumPy / PyTorch / DataLoader worker 的随机种子（`src/utils/seed.py`）；
* 训练/验证划分由固定种子的 `torch.Generator` 决定，多次运行完全一致；
* 每次运行输出 `metrics/config.json`，记录全部超参数、batch size 探测结果与设备信息；
* 每次运行输出 `metrics/metrics.csv`（一行式汇总表），多个实验的该文件可直接纵向拼接成对比表；
* 配合 `--run-tag` 实现**实验目录隔离**，避免多次实验互相覆盖或交叉引用；
* 需要完全确定性时可加 `--deterministic`（会略微降低速度）。

---

## 11. 引用

```bibtex
@inproceedings{liu2022convnet,
  title     = {A ConvNet for the 2020s},
  author    = {Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {11976--11986},
  year      = {2022}
}
```

* 论文：https://arxiv.org/abs/2201.03545
* 官方实现：https://github.com/facebookresearch/ConvNeXt

---

## 12. 已知问题与踩坑记录

### 12.1 fp16 AMP 梯度下溢 → 骨干网络被冻结（**已修复，务必知悉**）

| 项目 | 内容 |
|---|---|
| 触发条件 | GPU（T4 等不支持原生 bf16 的设备）+ `--amp`（默认开启） |
| 症状 | 日志**无任何报错**，但 `train_loss` 60 轮停在 ~2.2（ln10 = 2.303）、`val_acc` 峰值 ~26 %、测试 Top-1 ~25 % |
| 判据 1 | 日志 `AMP=True（autocast float16 + GradScaler）`，且每轮 `amp_scale` ≪ 65536（实测 0.0156） |
| 判据 2 | 检查点内 `scaler_state.scale = 0.0156`（默认 65536，相当于连续减半 22 次） |
| 判据 3 | 检查点内 `stages.1.*.pwconv1.weight` 的 `exp_avg_sq` **恒为 0**（一次都没更新过） |
| 机理 | LayerScale γ 初值 `1e-6` → 残差块内参数梯度 ≈ `1e-6`；fp16 动态范围过窄（最小次正规数 6e-8），梯度 × 小 scale 后被抹成精确的 0 |
| 影响 | 实际只有 stem + 分类头 + GroupNorm（+部分 γ）被训练，等价于浅层线性模型 → 25 %~30 % 的"假收敛" |
| 修复 | ① 默认优先 bf16；② 每轮输出 `amp_scale`；③ scale 崩塌时自动降级 fp32；④ 结果文件中记录 AMP 元信息（第 7.4 节） |
| 规避 | T4 上使用 `--no-amp`；或换 L4 / A100（自动 bf16）；或 `--no-amp-fallback` + 自行盯 `amp_scale` |
| 实证归档 | `05_results/_failed_runs/full_convnext_tiny_cifar10_amp_fp16_underflow/`（含完整证据链与曲线） |

### 12.2 显存自适应会"放大" batch size，等效 batch 可能远大于期望值

第 6 节的显存探测在显存富余时会**逐级上调** batch size（上限 `--max-batch-size`，默认 512）。
因此 `--batch-size 128` 在 T4 上实际跑成了 **512**（日志 `[batch-size] 最终确定 batch size = 512`）。
影响：学习率是按 128 标定的，等效 batch 变大后需要相应调大 lr 才能保持优化动态。
若希望严格固定 batch size（并让梯度累积接管），请加 `--no-auto-batch-size`。

### 12.3 一次运行是否可信：30 秒自查清单

1. `logs/history.csv`：`train_loss` 是否在下降？（60 轮仍停在 2.2 附近 → 不可信）
2. `logs/history.csv` 的 `amp_scale`：是否接近 65536？（≪ 1 → 见 12.1）
3. `metrics/test_metrics.json` 的 `best_val_epoch`：是否在训练后段？（停留在第 1~2 轮 → 不可信）
4. `metrics/metrics.csv` 的 `is_full_scale`：是否为 `True`？（`False` 表示用了快速子集，不能当作主结果）
5. `metrics/per_class_metrics.json` 的混淆矩阵列和：是否严重集中在少数类？（坍缩 → 不可信）

### 12.4 本地 Windows GPU（WDDM）：显存探测必须留余量

**现象**（RTX 4060 Laptop，8 GB，实测）：未修复前，显存的自动探测会把 batch size 一路放大到 512，
日志打印 `探针峰值显存 10856 MB，GPU 总显存 8.0 GB` —— 峰值超过物理显存说明它并没有真的"够用"；
随后真实训练在第一个前向就失败：`torch.OutOfMemoryError`，或（更常见、更隐蔽）
`RuntimeError: CUDA driver error: device not ready / Failed to create GPU mapping`。

**原因**：Windows 走 WDDM 显存管理，允许**显存过度提交**（超出物理显存的部分落到共享内存），
于是探针可以"成功"分配一块物理上放不下的显存，而真实训练（还要额外保存激活值、还要与桌面/浏览器
争夺显存）随即失败。Linux/CUDA 上没有这一行为，所以 Colab 上不容易遇到。

**已修复**（`src/utils/device.py`）：

1. 每次探针前 `empty_cache` + 重置峰值统计（旧实现不重置，导致峰值被历次探针累加而失真）；
2. 向上放大时要求 `探针峰值 ≤ 可用显存 × 70 %`，即**必须留出余量**，并打印拒绝原因；
3. 把 WDDM 的 `CUDA driver error / device not ready / Failed to create GPU mapping`
   一并视为"放不下"，从而**下调 batch size** 而不是让训练崩溃。

**实测效果**：修复后同一台机器上探测结果从 `512`（会崩）变为
`batch size = 128（探针峰值 2883 MB，可用显存 7000 MB）`，训练稳定运行。

**另建议**：在这类显存偏小、且与桌面共享显存的机器上，可以额外设置环境变量
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 以减少显存碎片；
若希望完全跳过探测、固定 batch size，可用 `--no-auto-batch-size`。

### 12.5 本地 Windows 长跑：DataLoader 共享内存提交耗尽（error 1455）

**现象**（RTX 4060 Laptop + 15.6 GB RAM，实测）：`--num-workers 4` 连续训练约 2 小时、
跑到第 33 轮时进程崩溃，栈顶为：

```
File ".../torch/utils/data/_utils/collate.py", line 273, in collate_tensor_fn
    storage = elem._typed_storage()._new_shared(numel, device=elem.device)
RuntimeError: Couldn't open shared file mapping: <torch_21920_...>, error code: <1455>
```

`1455` = `ERROR_COMMITMENT_LIMIT`（系统**提交内存/页面文件**不足），**不是**显存不足。

**原因**：Windows 上 PyTorch 的多进程 DataLoader 用**共享文件映射**把每个 batch 从 worker
搬运到主进程，而每次搬运约占 1.5 MB（= 128×3×32×32×4 B）的提交量；长时间运行（约 1.1 万个
batch）后会累积到系统提交上限。Linux 上用共享内存不会有这个量级的占用。

**规避**：

1. `--num-workers 0`：完全不走多进程搬运（本工程的数据集很小，单进程也够用；
   代价是每个 epoch 从约 2 分钟变成约 3.8 分钟）；
2. 若中途崩溃：用 `--resume auto` **原地续训**（模型/优化器/调度器/逐轮历史全部恢复，
   只损失当前一轮的进度），例如
   ```bash
   python train.py --run-tag <标签> --epochs 60 --model tiny --batch-size 128 \
                   --num-workers 0 --device cuda --resume auto ...
   ```
3. 监控方式：`logs/history.csv` 每轮落盘；若其行数长时间不增加，说明进程已崩溃（看 `logs/train.log` 最后一行即可定位）。
