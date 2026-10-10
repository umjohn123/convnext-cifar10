# 04_data —— 数据部分（dataset_info）

| 项目 | 内容 |
|---|---|
| 文档名称 | 数据集说明（Dataset Information） |
| **归属任务** | **T5 —— 数据集获取与准备** |
| 作业主题 | **AIassign2**（作业文件 `AIassign2.pdf`，正文标题 **AI Assignment #1**） |
| 算法 / 任务 | **ConvNeXt**（CVPR 2022） ／ 图像分类（CIFAR-10） |
| 文档版本 | v1.0 ｜ 编制日期：2026-10-09 |
| 归档位置 | `04_data/dataset_info.md` |

本目录是本作业的**数据部分**，存放训练/评估所用的数据集、划分索引与数据说明。
代码部分（`03_code/`）通过命令行参数 `--data-root` 读取本目录，二者引用关系明确、互不耦合。

---

## 1. 目录结构

```
04_data/
├── dataset_info.md            # 本文件：数据来源、划分协议、预处理、校验值、复现命令（T5 产物）
├── cifar10/                   # CIFAR-10 原始数据集（torchvision 标准缓存结构）
│   ├── cifar-10-python.tar.gz # 官方压缩包（170 MB，含完整 MD5 校验）
│   └── cifar-10-batches-py/   # 官方解压后的数据文件
│       ├── data_batch_1 ~ data_batch_5   # 训练集 5 个批次，各 10000 张
│       ├── test_batch                    # 测试集 1 个批次，10000 张
│       └── batches.meta                  # 类别名称等元信息
└── splits/                         # 数据划分索引（可复现性证据，由固定随机种子生成）
    ├── train_indices.pt            # 训练集索引（45000 个下标）
    ├── val_indices.pt              # 验证集索引（5000 个下标）
    ├── test_indices.pt             # 测试集**抽样排列** randperm(10000, seed=43)，用于派生测试子集
    ├── mvp_subset_train_indices.pt # MVP 小规模实验使用的训练索引（4500）
    ├── mvp_subset_val_indices.pt   # MVP 小规模实验使用的验证索引（500）
    ├── mvp_subset_test_indices.pt  # MVP 小规模实验使用的测试索引（2000）
    └── split_info.json             # 划分元信息：种子、比例、规模、对应文件
```

索引以 `torch.save(torch.tensor(..., dtype=torch.int64), ...)` 保存，
可用 `torch.load(path)` 读取；下标含义为「CIFAR-10 官方训练集/测试集内的位置」。

> **索引语义说明（重要）**
> * `train_indices.pt` / `val_indices.pt` 是训练/验证集的**实际迭代顺序**（已与代码逐项回归校验一致）；
> * `test_indices.pt` 是固定的**测试集抽样排列**（`randperm(10000, seed=43)`），仅用于派生测试子集；
>   **全量**测试评估按 torchvision 原生顺序 `0..9999` 迭代，因此该文件不是全量评估的迭代顺序；
> * 三个 `mvp_subset_*` 文件正是 `05_results/mvp_atto_subset5k` 实验实际使用的索引。
> 各文件的精确语义见 `splits/split_info.json` 的 `semantics` 字段。

---

## 2. 数据集来源

| 项目 | 内容 |
|---|---|
| 名称 | **CIFAR-10** |
| 提出者 | Alex Krizhevsky（2009），*Learning Multiple Layers of Features from Tiny Images* |
| 官方主页 | https://www.cs.toronto.edu/~kriz/cifar.html |
| 下载地址（Python 版） | https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz |
| 文件大小 | 170,498,071 字节（约 170 MB） |
| 许可 | 允许用于研究目的（citation required） |
| 引用 | Krizhevsky, A. *Learning Multiple Layers of Features from Tiny Images*. Tech. Report, 2009. |

### 2.1 数据规模与内容

| 项目 | 数值 |
|---|---|
| 总图像数 | 60,000 |
| 图像尺寸 | 32 × 32 像素，RGB 三通道 |
| 类别数 | 10（每类 6,000 张） |
| 官方划分 | 训练 50,000 / 测试 10,000 |
| 类别名称 | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |

### 2.2 官方 MD5 校验值（完整性验证）

| 文件 | MD5 |
|---|---|
| `cifar-10-python.tar.gz` | `c58f30108f718f92721af3b95e74349a` |
| `data_batch_1` | `c99cafc152244af753f735de768cd75f` |
| `data_batch_2` | `d4bba439e000b95fd0a9bffe97cbabec` |
| `data_batch_3` | `54ebc0e0eda56f17810534c785d663d4` |
| `data_batch_4` | `2d6fae7d3d2dc13fd8464dd1eeb2b0b6` |
| `data_batch_5` | `b3a24a1c40ba3d15998b40c90d30f0cc` |
| `test_batch` | `40351d587109b95175f43aff81a1287e` |
| `batches.meta` | `5ff9c542aee3614f3951f8cda6e48888` |

> 代码在首次运行时由 `torchvision.datasets.CIFAR10` 自动完成下载与 MD5 校验，
> 校验失败会自动重新下载，避免使用损坏数据。

---

## 3. 本工程的数据划分协议

CIFAR-10 自带的 10,000 张测试集**不参与任何训练与调参**，用作最终评估。
为避免在 50,000 张训练集上"自测自选"，代码按**固定随机种子**从中划出验证集：

| 子集 | 规模 | 用途 | 是否做数据增强 |
|---|---|---|---|
| 训练集 | 45,000 | 更新模型参数 | ✅ 是 |
| 验证集 | 5,000 | 每轮监控、挑选最优 checkpoint | ❌ 否（确定性） |
| 测试集 | 10,000（官方） | 最终一次性评估 | ❌ 否（确定性） |

**划分可复现机制**：使用 `torch.Generator().manual_seed(seed)` 生成 `torch.randperm(50000)`，
取前 45,000 为训练集、后 5,000 为验证集；`seed` 由 `--seed` 指定（默认 `42`）。
该划分结果已导出到 `splits/` 目录作为可复现性证据。

### 3.1 快速验证子集模式（可选）

在 CPU 等算力受限环境下，可用 `--train-subset-size` / `--test-subset-size`
从同一固定种子的排列中截取更小的子集，用于快速跑通流程（对应执行计划中的 I0/MVP 实验）。
该模式会在日志中打印**显式警告**，并在结果文件中标记 `is_full_scale: false`。

---

## 4. 预处理与数据增强

### 4.1 归一化参数

由 CIFAR-10 训练集在 `[0, 1]` 尺度下统计得到：

```
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
```

### 4.2 变换流水线

| 阶段 | 变换 |
|---|---|
| 训练 | `RandomCrop(32, padding=4, padding_mode="reflect")` → `RandomHorizontalFlip(p=0.5)` → `ToTensor()` → `Normalize(mean, std)` → `RandomErasing(p=0.25, scale=(0.02,0.2), value=0)` |
| 验证 / 测试 | `ToTensor()` → `Normalize(mean, std)`（完全确定性） |
| 可选 | `RandAugment(num_ops=2, magnitude=9)`（默认关闭，用 `RandomAugment` 开关控制） |

> 训练集与验证集使用**同一个索引划分**，但分别绑定"带增强"与"不带增强"的变换对象，
> 因此验证指标不受随机增强干扰，可作为模型选择的可靠依据。

---

## 5. 如何获取数据（复现步骤）

数据无需手动下载，代码首次运行时会自动完成。也可单独执行：

```bash
cd 03_code
python -c "from torchvision import datasets; datasets.CIFAR10(root='../04_data/cifar10', train=True, download=True); datasets.CIFAR10(root='../04_data/cifar10', train=False, download=True)"
```

校验数据完整性：

```bash
python -c "import hashlib; print(hashlib.md5(open('../04_data/cifar10/cifar-10-python.tar.gz','rb').read()).hexdigest())"
# 期望输出：c58f30108f718f92721af3b95e74349a
```

### 5.0 下载加速（推荐）

官方源（`www.cs.toronto.edu/~kriz`）**单连接**在部分网络环境下速率很低：
本机实测仅 **0.11 MB/s**，完整下载 162 MB 需要约 **25 分钟**。
该源**支持 HTTP Range**，因此可用多连接分片把聚合带宽提升一个量级：

| 方式 | 实测速率 | 162 MB 下载耗时 |
|---|---|---|
| 单连接（torchvision 默认） | 0.11 MB/s | 约 25 分钟 |
| **多连接分块**（本工程的 `scripts/fetch_cifar10.py`） | **1.3 ~ 8 MB/s**（随镜像与网络波动） | **约 20 ~ 120 秒**（含 MD5 校验与解压） |

```bash
cd 03_code
python scripts/fetch_cifar10.py --dest ../04_data/cifar10                 # 下载 + 校验 + 解压（默认 8 连接 × 4 MiB）
python scripts/fetch_cifar10.py --dest ../04_data/cifar10 --probe-only    # 只体检镜像、不下载
python scripts/fetch_cifar10.py --dest ../04_data/cifar10 --threads 16    # 加大并发
```

脚本会：镜像体检（`HEAD` 查远端长度 + 采样测速 + 探测 Range）→ 排除长度不符的**损坏镜像**
→ 多连接分块下载（断点续传、指数退避重试、失败分块单独重下）→ 校验文件大小 + MD5
→ 自动解压；任何一步失败都会自动回退（降并发 → 单连接 → 换镜像）。
若只是想在无网络环境下验证流程，可直接使用 `--dataset synthetic`（本地合成数据，零下载）。

### 5.1 重新生成划分索引（可选）

`splits/` 下的索引由固定随机种子生成，可随时重算以校验一致性：

```bash
cd 03_code
python -c "
import torch, os
os.makedirs('../04_data/splits', exist_ok=True)
g = torch.Generator().manual_seed(42)
perm = torch.randperm(50000, generator=g).tolist()
torch.save(torch.tensor(perm[:45000]), '../04_data/splits/train_indices.pt')
torch.save(torch.tensor(perm[45000:]), '../04_data/splits/val_indices.pt')
g2 = torch.Generator().manual_seed(43)
torch.save(torch.tensor(torch.randperm(10000, generator=g2).tolist()), '../04_data/splits/test_indices.pt')
print('done')
"
```

> 划分逻辑与 `03_code/src/data/cifar10.py::build_datasets()` 完全一致：
> 训练池用 `seed=--seed`（默认 42）的 `randperm(50000)`，测试集用 `seed+1` 的 `randperm(10000)`；
> 快速验证子集则取训练池排列的前 `--train-subset-size` 个元素。

---

## 6. 与代码 / 结果的引用关系

| 引用关系 | 方式 | 说明 |
|---|---|---|
| 结果 → 数据 | `05_results/<实验标签>/metrics/config.json` 中的 `data_root`、`train_subset_size`、`test_subset_size`、`val_ratio` | 记录本次实验实际使用的数据位置与规模（每个实验独立成目录） |
| 结果 → 数据划分 | `04_data/splits/split_info.json` + `*.pt` 索引文件 | 结果可据此完全重建数据划分；`mvp_subset_*` 三个索引对应 `05_results/mvp_atto_subset5k/` |
| 代码 → 数据 | 命令行参数 `--data-root`（作业结构下为 `../04_data/cifar10`） | 代码不硬编码数据路径 |

---

## 7. 版本控制建议

数据集体积较大（压缩包 170 MB + 解压后约 180 MB），若托管到 Git 平台，
建议在 `.gitignore` 中排除二进制数据文件、仅保留本说明与 `splits/` 索引：

```gitignore
04_data/cifar10/cifar-10-python.tar.gz
04_data/cifar10/cifar-10-batches-py/
```
