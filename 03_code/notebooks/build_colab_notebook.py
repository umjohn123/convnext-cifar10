# -*- coding: utf-8 -*-
"""
由本工程源码**自动生成**自包含的 Google Colab Notebook。

为什么需要这个脚本
------------------
报告要求"代码可在 Google Colab 中直接运行"。为了让 Notebook 与工程源码**永远保持一致**，
这里不手写 Notebook，而是读取 ``03_code/`` 下的真实源码文件，
逐个生成为 ``%%writefile`` 单元格，最后追加"训练 / 评估 / 展示结果"单元格。

这样做的收益：
* Notebook 内容 = 工程源码，不存在"两份代码不一致"的风险；
* 修改源码后重新执行本脚本即可刷新 Notebook。

用法::

    cd 03_code
    python notebooks/build_colab_notebook.py
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(NOTEBOOK_DIR)
OUTPUT_PATH = os.path.join(NOTEBOOK_DIR, "ConvNeXt_CIFAR10_Colab.ipynb")

# Colab 中的工程根目录
COLAB_PROJECT_DIR = "/content/convnext_cifar10"

# 源码文件的**优先写入顺序**表（保证阅读顺序符合依赖关系）。
#
# 注意：本表只是一个"排序基准"，**不是**最终写入的文件集合。
# 真正写入 Notebook 的文件由 discover_source_files() 决定 ——
# 它会自动补入工程中其余所有的 .py 文件，从而避免"新增模块后忘记登记"，
# 否则 Colab 上会出现 ModuleNotFoundError（例如新增 src/utils/tta.py 却漏登记）。
SOURCE_FILES: List[str] = [
    "configs.py",
    "src/__init__.py",
    "src/utils/__init__.py",
    "src/utils/logger.py",
    "src/utils/seed.py",
    "src/utils/misc.py",
    "src/utils/csv_logger.py",
    "src/utils/amp.py",
    "src/utils/device.py",
    "src/utils/mixup.py",
    "src/utils/tta.py",
    "src/utils/visualize.py",
    "src/data/__init__.py",
    "src/data/cifar10.py",
    "src/models/__init__.py",
    "src/models/convnext.py",
    "src/engine/__init__.py",
    "src/engine/evaluator.py",
    "src/engine/trainer.py",
    "train.py",
    "evaluate.py",
]

# 不写入 Notebook 的**顶层**目录（只按路径第一段匹配，因此不会误伤 src/data 这类子包）
EXCLUDED_TOP_DIRS = {".git", ".ipynb_checkpoints", "__pycache__", "notebooks", "outputs", "data"}
# 不写入 Notebook 的文件名前缀（临时脚本等）
EXCLUDED_FILE_PREFIXES = ("_tmp", ".")


def discover_source_files() -> List[str]:
    """收集需要写入 Notebook 的**全部**源码文件（自动发现 + 显式排序）。

    实现
    ----
    1. 遍历工程根目录，收集所有 ``.py`` 文件（跳过 :data:`EXCLUDED_TOP_DIRS` 与临时文件）；
    2. 先按 :data:`SOURCE_FILES` 的显式顺序排列已登记的文件；
    3. 再把**未登记**的其余文件按路径顺序追加到末尾。

    这样即使日后新增模块而忘记更新 ``SOURCE_FILES``，它也会被自动包含进 Notebook，
    从根本上消除"Colab 缺少模块"这一类问题（缺失只会影响可读顺序，不会导致 ImportError）。
    """
    discovered: List[str] = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        relative_root = os.path.relpath(root, PROJECT_ROOT).replace(os.sep, "/")
        top_level = relative_root.split("/")[0] if relative_root != "." else ""
        if top_level in EXCLUDED_TOP_DIRS:
            dirs[:] = []          # 不再向下遍历
            continue
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".ipynb_checkpoints"))

        for name in sorted(files):
            if not name.endswith(".py") or name.startswith(EXCLUDED_FILE_PREFIXES):
                continue
            discovered.append(f"{relative_root}/{name}" if relative_root != "." else name)

    ordered = [path for path in SOURCE_FILES if path in discovered]
    ordered += sorted(path for path in discovered if path not in ordered)
    return ordered


def extract_embedded_paths(notebook: dict) -> List[str]:
    """从 Notebook 中提取所有 ``%%writefile`` 单元格对应的文件路径（按出现顺序）。"""
    paths: List[str] = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("%%writefile "):
            paths.append(source.split("\n", 1)[0][len("%%writefile "):].strip())
    return paths


def verify_notebook(path: str) -> List[str]:
    """自检生成的 Notebook 是否**完整且与工程源码一致**，返回问题清单（空列表表示通过）。

    检查三项（前两项即此前遗漏的"完整性"校验）：
    1. **完整性**：工程里每个源码文件都必须出现在 Notebook 中；
    2. **无多余**：Notebook 中不能有工程里不存在的文件；
    3. **逐字一致**：每个 ``%%writefile`` 单元格的内容与磁盘上的源码相同。
    """
    problems: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        notebook = json.load(fh)

    embedded = extract_embedded_paths(notebook)
    expected = discover_source_files()

    for missing in sorted(set(expected) - set(embedded)):
        problems.append(f"缺失：工程文件 {missing} 未写入 Notebook")
    for extra in sorted(set(embedded) - set(expected)):
        problems.append(f"多余：Notebook 含工程不存在的文件 {extra}")

    embedded_content = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not source.startswith("%%writefile "):
            continue
        if "\n" not in source:
            continue
        header, body = source.split("\n", 1)
        rel = header[len("%%writefile "):].strip()
        embedded_content[rel] = body.strip()

    for rel in sorted(set(expected) & set(embedded)):
        with open(os.path.join(PROJECT_ROOT, rel), "r", encoding="utf-8") as fh:
            if fh.read().strip() != embedded_content.get(rel, None):
                problems.append(f"内容不一致：{rel}")
    return problems

# 每个文件所属的模块说明（用于 Markdown 小节标题）
SECTION_OF: Dict[str, str] = {
    "configs.py": "2. 全局配置",
    "src/utils/logger.py": "3. 工具模块 —— 日志",
    "src/utils/seed.py": "3. 工具模块 —— 随机种子",
    "src/utils/misc.py": "3. 工具模块 —— 指标累加与 JSON",
    "src/utils/csv_logger.py": "3. 工具模块 —— CSV 导出（history.csv / metrics.csv）",
    "src/utils/amp.py": "3. 工具模块 —— 混合精度 AMP",
    "src/utils/device.py": "3. 工具模块 —— 设备与显存自适应 batch size",
    "src/utils/mixup.py": "3. 工具模块 —— Mixup / CutMix",
    "src/utils/tta.py": "3. 工具模块 —— 测试时增强（TTA）",
    "src/utils/visualize.py": "3. 工具模块 —— 训练曲线与混淆矩阵绘图",
    "src/data/cifar10.py": "4. 数据模块 —— CIFAR-10 常量、变换与数据底座",
    "src/data/synthetic.py": "4. 数据模块 —— 合成数据（零下载冒烟测试用）",
    "src/data/loader.py": "4. 数据模块 —— 划分协议与 DataLoader 组装",
    "src/models/convnext.py": "5. 模型模块 —— ConvNeXt（CVPR 2022）",
    "src/engine/evaluator.py": "6. 评估模块 —— Top-k 准确率 / 逐类指标 / 混淆矩阵",
    "src/engine/trainer.py": "7. 训练模块 —— 训练循环与检查点管理",
    "train.py": "8. 训练入口脚本",
    "evaluate.py": "9. 独立评估入口脚本",
    "scripts/fetch_cifar10.py": "10. 工具脚本 —— CIFAR-10 高速下载器（镜像体检 + 多连接分块）",
}


# --------------------------------------------------------------------------- #
# Notebook 单元格构造
# --------------------------------------------------------------------------- #
def md_cell(text: str) -> dict:
    """构造 Markdown 单元格。"""
    text = text.strip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str, title: str = "") -> dict:
    """构造代码单元格；``title`` 非空时添加 ``#@title`` 折叠标题（便于 Colab 使用）。"""
    text = text.strip("\n") + "\n"
    if title:
        text = f"#@title {title}\n{text}"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"cellView": "form"} if title else {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def writefile_cell(relative_path: str, content: str) -> dict:
    """把源码文件包装成 ``%%writefile`` 单元格。"""
    text = f"%%writefile {relative_path}\n{content}"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# --------------------------------------------------------------------------- #
# 主体单元格内容
# --------------------------------------------------------------------------- #
TITLE_MD = """
# ConvNeXt on CIFAR-10 — 可在 Google Colab 直接运行的完整实现

**课程**：CISC3024 Pattern Recognition ｜ **作业**：AIassign2（AI Assignment #1）

**算法**：ConvNeXt —— *A ConvNet for the 2020s*, **CVPR 2022**（arXiv:2201.03545）
｜ **类别**：Deep CNN ｜ **应用**：计算机视觉 / 模式识别（图像分类）


## 本 Notebook 如何使用

1. 建议先在 Colab 中开启 GPU：**代码执行程序 → 更改运行时类型 → T4 GPU**；
2. 从上到下依次执行单元格：第 1 步做环境自检，第 2–10 步会**自动生成完整工程源码**，
   第 11 步（可选）跑快速冒烟测试，第 12 步开始正式训练，之后自动展示曲线与指标；
3. 所有源码与工程结构完全一致，产物写入 `outputs/`（模型、日志、图表、指标）。

## 流水线总览

```
CIFAR-10 下载/划分/增强  ->  ConvNeXt 构建  ->  显存自适应 batch size 探测
        ->  训练(warmup+cosine, AdamW, AMP, Mixup/CutMix, 梯度累积)
        ->  每轮验证并保存最优检查点  ->  测试集评估(Top-1/Top-5, 逐类准确率, 混淆矩阵)
        ->  绘制损失曲线 / 准确率曲线 / 混淆矩阵 / 逐类准确率
```
"""

ENV_CHECK_CODE = """
import os, sys, platform
import torch
import torchvision

print("=" * 80)
print("Python       :", sys.version.split()[0], "|", platform.platform())
print("PyTorch      :", torch.__version__)
print("torchvision  :", torchvision.__version__)
print("NumPy        :", __import__("numpy").__version__)
print("-" * 80)
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("CUDA         : 可用")
    print("GPU          :", torch.cuda.get_device_name(0))
    print(f"显存总量      : {props.total_memory / 1024 ** 3:.1f} GB | 计算能力 sm_{props.major}{props.minor}")
else:
    print("CUDA         : 不可用 —— 将在 CPU 上训练（速度较慢）")
    print("提示：代码执行程序 -> 更改运行时类型 -> 选择 T4 GPU")
print("=" * 80)
"""

SETUP_CODE = f"""
import os

PROJECT_DIR = "{COLAB_PROJECT_DIR}"
SUBDIRS = [
    "src/data", "src/models", "src/engine", "src/utils", "notebooks", "scripts",
    "outputs/figures", "outputs/logs", "outputs/models", "outputs/metrics",
]

os.makedirs(PROJECT_DIR, exist_ok=True)
for sub in SUBDIRS:
    os.makedirs(os.path.join(PROJECT_DIR, sub), exist_ok=True)

# 切换到工程根目录（后续 !python train.py 都在此目录执行）
os.chdir(PROJECT_DIR)
print("工作目录 :", os.getcwd())
print("已创建子目录 :", SUBDIRS)
"""

SMOKE_TEST_CODE = """
import torch

# 冒烟测试的唯一目标是：确认"环境 + 代码链路"通畅。
# 为此它做了两件事：
#   1) 使用**本地生成的合成数据**（--dataset synthetic）—— 完全不下载任何东西。
#      原先生成器的冒烟测试会触发 CIFAR-10 下载（162 MB）。官方源单连接在部分网络下
#      只有约 0.1 MB/s，仅下载就要 25 分钟，这是它无法在 2 分钟内完成的首要原因。
#   2) 用 --max-steps-per-epoch 给训练量设上界 —— 总耗时只取决于 step 数，
#      与数据集规模、硬件性能解耦（Colab 纯 CPU 运行时仅 2 vCPU，这点很关键）。
GPU = torch.cuda.is_available()
if GPU:
    MODEL, BATCH, TRAIN_N, TEST_N, STEPS, EPOCHS = "atto", 128, 2048, 1024, 20, 2
else:
    MODEL, BATCH, TRAIN_N, TEST_N, STEPS, EPOCHS = "atto", 32, 512, 256, 4, 2

print(f"运行时={'GPU' if GPU else 'CPU'} | model={MODEL} batch={BATCH} "
      f"train={TRAIN_N} test={TEST_N} steps/epoch={STEPS} epochs={EPOCHS}")

!python train.py --run-tag smoke_test --dataset synthetic --synthetic-train-size {TRAIN_N} --synthetic-test-size {TEST_N} --model {MODEL} --batch-size {BATCH} --max-steps-per-epoch {STEPS} --epochs {EPOCHS} --exp-name smoke_test
"""

FETCH_DATA_CODE = """
# （可选）提前把 CIFAR-10 下载到工作目录，供正式训练使用。
#
# 为什么要单独跑这一步：官方源单连接在部分网络环境下只有约 0.1 MB/s，
# 162 MB 需要约 25 分钟；而该源支持 HTTP Range，因此脚本用**多连接分块**
# 把聚合带宽提升一个量级（实测 1.3 ~ 8 MB/s，下载+校验+解压约 20 ~ 120 秒）。
# 脚本会自动：镜像体检（剔除被截断的损坏镜像）-> 分块并行下载（断点续传、指数退避）
# -> 校验文件大小 + MD5 -> 解压；任何一步失败都会自动回退（降并发 -> 单连接 -> 换镜像）。
#
# 参数说明（默认值已按 Colab 调优，通常无需改动）：
#   --threads 8      并发连接数：8 已接近多数镜像的饱和点，16 容易触发限流且收益递减
#   --chunk-mb 4     单个分块大小：太小放大请求开销，太大则单块失败的重下代价高
#   --max-retries 8  单分块最大重试次数（配合指数退避，上限 30s）
!python scripts/fetch_cifar10.py --dest ./data --threads 8 --chunk-mb 4

# 只体检镜像、不下载（想先看哪个镜像可用、速率多快时用）：
# !python scripts/fetch_cifar10.py --dest ./data --probe-only

# 若某个镜像分块反复失败，脚本会自动降并发/改单连接/换镜像；也可手动指定镜像：
# !python scripts/fetch_cifar10.py --dest ./data --threads 4 --mirrors http://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz

# 想跨会话复用、避免每次重新下载？把数据放到 Google Drive 再挂载即可：
#   from google.colab import drive; drive.mount('/content/drive')
#   !python scripts/fetch_cifar10.py --dest /content/drive/MyDrive/cifar10
#   之后训练时加  --data-root /content/drive/MyDrive/cifar10
"""

TRAIN_CODE = """
# 正式训练：ConvNeXt-Tiny（论文 ConvNeXt-T 的结构）+ 60 epochs
#   --batch-size 128 只是"期望值"，显存不足会自动下调并用梯度累积补偿（见日志 [batch-size]）
#   --stem-mode conv 为 CIFAR-10 小分辨率的适配版 Stem（3x3, stride=1）
#   --dataset 默认 CIFAR10（真实数据）；若已执行过 3-B，数据已缓存，不会重复下载
#   --run-tag 指定实验标签 -> 全部产物写入 outputs/<标签>/，多次实验互不覆盖
#
#   --no-amp：**T4 上必须加**。原因：本模型 LayerScale 初值为 1e-6，18 个残差块叠加后
#             浅层 Block 的真实梯度只有 1e-6 甚至更低；fp16 的最小次正规数是 6e-8，
#             一旦 GradScaler 的动态 scale 因 fp16 溢出被压到 0.0156 量级，
#             梯度乘上 scale 后仍会下溢成**精确的 0**，这些参数从此不再更新
#             （实测后果：60 轮 loss 不动、测试精度只有 25.7%，是"假收敛"）。
#             fp32 没有这个动态范围问题。若换到 L4/A100（原生支持 bf16），
#             程序会自动改用 bfloat16，此时可以去掉 --no-amp（bf16 与 fp32 同阶码范围）。
#
#   自检方法：看每轮日志末尾的 amp_scale，正常应稳定在 65536 附近；
#             若持续 < 1，说明 fp16 已在丢梯度（程序会自动降级为 fp32 并打印告警）。
#
#   --resume auto：若本次实验目录下已有 last.pt，则自动续训（Colab 断线重连专用：
#                 重连后**重新执行本单元格**即可接着训练，不会从头再来）
#
# 若想更快看到结果（T4 fp32 更慢，可换更小的模型/更少的轮数）：
#   RUN_TAG = "pico_30e"
#   !python train.py --run-tag {RUN_TAG} --epochs 30 --model pico --batch-size 128 --no-amp
RUN_TAG = "full_convnext_tiny_cifar10"
!python train.py --run-tag {RUN_TAG} --epochs 60 --model tiny --batch-size 128 --stem-mode conv --no-amp

# 若中途断连，只需把上一行改成带 --resume auto 的版本再执行一次：
# !python train.py --run-tag {RUN_TAG} --epochs 60 --model tiny --batch-size 128 --stem-mode conv --no-amp --resume auto
"""

SHOW_RESULTS_CODE = """
import json, os
from IPython.display import Image, display, Markdown

# 与训练单元格保持一致的实验标签；若改了标签，这里同步修改即可
# （只跑了 3-A 冒烟测试时，把下面改成 "smoke_test" 即可查看其结果）
RUN_TAG = "full_convnext_tiny_cifar10"
RUN_DIR = os.path.join("outputs", RUN_TAG)

FIG_DIR = os.path.join(RUN_DIR, "figures")
RESULT_FILE = os.path.join(RUN_DIR, "metrics", "test_metrics.json")
CSV_FILE = os.path.join(RUN_DIR, "metrics", "metrics.csv")

print("实验目录 :", os.path.abspath(RUN_DIR))
if os.path.isdir(RUN_DIR):
    for root, _dirs, files in os.walk(RUN_DIR):
        for name in sorted(files):
            p = os.path.join(root, name)
            print(f"  {p}  ({os.path.getsize(p):,} B)")

# ---- 1) 训练曲线：损失 / 准确率 / 学习率 ---- #
curves = os.path.join(FIG_DIR, "training_curves.png")
if os.path.isfile(curves):
    display(Markdown("### 训练曲线（损失 / 准确率 / 学习率）"))
    display(Image(filename=curves))
else:
    print("尚未生成训练曲线，请先运行训练单元格。")

# ---- 2) 混淆矩阵与逐类准确率 ---- #
for fname, caption in [
    ("confusion_matrix.png", "### 测试集混淆矩阵（按真实类别归一化）"),
    ("per_class_accuracy.png", "### 测试集逐类准确率"),
    ("data_samples.png", "### CIFAR-10 数据样例"),
]:
    path = os.path.join(FIG_DIR, fname)
    if os.path.isfile(path):
        display(Markdown(caption))
        display(Image(filename=path))

# ---- 3) 最终测试指标 ---- #
if os.path.isfile(RESULT_FILE):
    with open(RESULT_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)
    display(Markdown("### 最终测试指标"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

# ---- 4) 指标汇总表（metrics.csv）---- #
if os.path.isfile(CSV_FILE):
    display(Markdown("### 指标汇总表（metrics.csv）"))
    print(open(CSV_FILE, "r", encoding="utf-8").read())
"""

EVALUATE_CODE = """
# 用训练得到的验证集最优模型，独立对测试集再评估一次（不重新训练）。
# 脚本会自动从检查点内的 config 恢复模型结构，因此无需再指定 --model / --stem-mode。
RUN_TAG = "full_convnext_tiny_cifar10"

# 常规评估（不启用 TTA）
!python evaluate.py --ckpt ./outputs/{RUN_TAG}/models/best.pt

# 启用测试时增强（原图 + 水平翻转，平均 softmax 概率）：耗时约翻倍，通常 +0.2~0.5 个百分点的 Top-1
!python evaluate.py --ckpt ./outputs/{RUN_TAG}/models/best.pt --tta
"""

DOWNLOAD_CODE = """
import os, shutil

archive_base = "/content/convnext_cifar10_outputs"
zip_path = shutil.make_archive(archive_base, "zip", "outputs")
print("已打包 :", zip_path, f"({os.path.getsize(zip_path) / 1024 ** 2:.1f} MB)")

try:
    from google.colab import files
    files.download(zip_path)      # 仅在 Colab 中可用
except Exception as err:
    print("（非 Colab 环境，已跳过自动下载）:", err)
"""

FOOTER_MD = """
---

## 说明与对应关系

### 与模块化工程的一致性

本 Notebook 的第 2–10 步会用 `%%writefile` 生成与 `03_code/` **完全相同**的源码文件，
因此 Notebook 内运行的结果与命令行运行 `python train.py` 完全一致：

| 模块 | 文件 | 职责 |
|---|---|---|
| 配置 | `configs.py` | 数据 / 模型 / 训练超参数与命令行解析 |
| 数据 | `src/data/cifar10.py` | CIFAR-10 常量、变换与数据底座 |
| 数据 | `src/data/synthetic.py` | 合成数据（零下载冒烟测试用） |
| 数据 | `src/data/loader.py` | 划分协议、`TransformSubset` 与 DataLoader 组装 |
| 数据工具 | `scripts/fetch_cifar10.py` | CIFAR-10 高速下载器（镜像体检 + 多连接分块 + MD5 校验 + 回退链） |
| 模型 | `src/models/convnext.py` | ConvNeXt 定义（倒置瓶颈、7x7 深度卷积、LayerNorm、Layer Scale、DropPath） |
| 训练 | `src/engine/trainer.py` | 训练循环、AdamW、warmup+cosine、AMP、梯度累积、检查点 |
| 评估 | `src/engine/evaluator.py` | Top-1/Top-5、逐类准确率、混淆矩阵（支持 TTA 口径） |
| 测试时增强 | `src/utils/tta.py` | 原图 + 水平翻转两路 TTA，平均 softmax 概率 |
| CSV 导出 | `src/utils/csv_logger.py` | `history.csv`（逐轮）与 `metrics.csv`（结果汇总） |
| 可视化 | `src/utils/visualize.py` | 损失曲线、准确率曲线、混淆矩阵、逐类准确率 |
| 显存自适应 | `src/utils/device.py` | 探测可用 batch size，不足时自动下调并可用梯度累积补偿 |

### 输出产物

```
outputs/
└── <run-tag>/               # 每次实验一个独立目录（由 --run-tag 指定，互不覆盖）
    ├── figures/   【图表】 training_curves.png, confusion_matrix.png,
    │                      per_class_accuracy.png, data_samples.png
    ├── logs/      【日志】 train.log, evaluate.log, history.json, history.csv
    ├── models/    【模型】 best.pt, last.pt
    └── metrics/   【指标】 config.json, metrics.csv, test_metrics.json,
                           per_class_metrics.json, eval_metrics.json
```

### 参考文献

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie.
*A ConvNet for the 2020s.* CVPR 2022, pp. 11976–11986. arXiv:2201.03545.
官方实现：https://github.com/facebookresearch/ConvNeXt
"""


# --------------------------------------------------------------------------- #
# 构建 Notebook
# --------------------------------------------------------------------------- #
def build() -> str:
    """生成 Notebook 文件并返回其路径。"""
    cells: List[dict] = []

    # ---- 标题与总览 ---- #
    cells.append(md_cell(TITLE_MD))

    # ---- 0. 环境自检 ---- #
    cells.append(md_cell("## 1. 环境自检\n\n确认 Colab 的 Python / PyTorch 版本与 GPU 是否可用。"))
    cells.append(code_cell(ENV_CHECK_CODE, title="1. 环境自检（版本 / GPU / 显存）"))

    # ---- 1. 创建工作目录 ---- #
    cells.append(md_cell(
        f"## 2. 创建工作目录\n\n"
        f"在 Colab 中于 `{COLAB_PROJECT_DIR}` 下创建完整的工程目录结构。"
    ))
    cells.append(code_cell(SETUP_CODE, title="2. 创建工作目录与子目录"))

    # ---- 2. 写入全部源码 ---- #
    cells.append(md_cell(
        "## 3. 生成工程源码\n\n"
        "以下每个单元格都会把一个源码文件写入磁盘（`%%writefile`），"
        "内容与 `03_code/` 工程完全一致。依次执行即可。"
    ))

    current_section = None
    for relative_path in discover_source_files():
        absolute_path = os.path.join(PROJECT_ROOT, relative_path)
        if not os.path.isfile(absolute_path):
            raise FileNotFoundError(f"缺少源码文件：{absolute_path}")

        section = SECTION_OF.get(relative_path)
        if section and section != current_section:
            cells.append(md_cell(f"### {section}"))
            current_section = section

        with open(absolute_path, "r", encoding="utf-8") as f:
            content = f.read()

        cells.append(md_cell(f"`{relative_path}`（{len(content.splitlines())} 行）"))
        cells.append(writefile_cell(relative_path, content))

    # ---- 3. 执行区域 ---- #
    cells.append(md_cell(
        "## 4. 运行\n\n"
        "工程源码已全部就绪。建议按下面的顺序执行；其中 3-B 与 3-C 是**唯一需要联网**的步骤，\n"
        "而 3-A 冒烟测试**完全不需要联网**，可在几十秒内确认环境是否正常。\n\n"
        "| 步骤 | 内容 | 预计耗时 | 需要联网 |\n"
        "|---|---|---|---|\n"
        "| 3-A | 快速冒烟测试（本地合成数据 + 受限 step，**零下载**） | GPU 约 30 s / CPU 约 1 min | 不需要 |\n"
        "| 3-B | （可选）下载并缓存 CIFAR-10（多连接加速 + MD5 校验） | 约 2 分钟（视网络） | 需要 |\n"
        "| 3-C | 正式训练（ConvNeXt-Tiny, 60 epochs, `--no-amp`） | T4 上约 3~5 小时（fp32） | 首次需要 |\n"
        "| 3-D | 展示训练曲线与评估指标 | 秒级 | 不需要 |\n"
        "| 3-E | （可选）独立评估测试集 | 数十秒 | 不需要 |\n"
        "| 3-F | （可选）打包下载全部产物 | 秒级 | 不需要 |\n"
        "\n"
        "> **T4 上请勿使用 fp16 AMP**：本模型的 LayerScale 初值为 `1e-6`，浅层 Block 的梯度量级只有 `1e-6`，\n"
        "> 一旦 GradScaler 的 scale 被 fp16 溢出压到 `0.0156` 这类小值，梯度会下溢为精确的 0，\n"
        "> 训练表现为「loss 不下降、精度停在 ~25%」的假收敛（详见 `src/utils/amp.py` 的模块说明）。\n"
        "> 3-C 默认已加 `--no-amp`；换到 L4/A100 时可去掉（程序会自动使用 bfloat16）。\n"
        "\n"
        "> **在本机 GPU 上运行（Windows / Linux，非 Colab）**：先装 CUDA 版 PyTorch（命令见\n"
        "> `requirements.txt`），然后把命令换成 `python train.py ... --device cuda`（`--device auto` 也会自动选到 CUDA）。\n"
        "> 注意事项：① Ampere 及以后（RTX 30/40、L4、A100）程序自动用 **bfloat16**，T4 等回退 fp16（建议加 `--no-amp`）；\n"
        "> ② Windows 长跑建议 `--num-workers 0`（多进程 DataLoader 的共享内存提交可能耗尽，报 `error 1455`）；\n"
        "> ③ 显存较小的卡依赖自动探测即可 —— 探针要求峰值 ≤ 可用显存 70%，会为训练留出余量；\n"
        "> ④ 若长跑中断，用 `--resume auto` 原地续训。详见 `03_code/README.md` 第 2.2 / 6 / 12.4 / 12.5 节。\n"
    ))
    cells.append(code_cell(SMOKE_TEST_CODE, title="3-A. 快速冒烟测试（零下载，约 30 秒 ~ 1 分钟）"))
    cells.append(code_cell(FETCH_DATA_CODE, title="3-B. （可选）下载并缓存 CIFAR-10（多连接加速）"))
    cells.append(code_cell(TRAIN_CODE, title="3-C. 正式训练（ConvNeXt-Tiny, 60 epochs）"))
    cells.append(code_cell(SHOW_RESULTS_CODE, title="3-D. 展示训练曲线与评估指标"))
    cells.append(code_cell(EVALUATE_CODE, title="3-E. （可选）独立评估测试集"))
    cells.append(code_cell(DOWNLOAD_CODE, title="3-F. （可选）打包下载全部产物"))

    # ---- 4. 页脚 ---- #
    cells.append(md_cell(FOOTER_MD))

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True,
                "accelerator": "GPU",       # 提示 Colab 默认使用 GPU 运行时
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
                "language": "python",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    os.makedirs(NOTEBOOK_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write("\n")

    return OUTPUT_PATH


if __name__ == "__main__":
    path = build()
    print(f"已生成 Notebook：{path}")

    with open(path, "r", encoding="utf-8") as fh:
        notebook = json.load(fh)
    n_code = sum(1 for cell in notebook["cells"] if cell["cell_type"] == "code")
    n_md = sum(1 for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    print(f"单元格总数 = {len(notebook['cells'])}（代码 {n_code} 个 / Markdown {n_md} 个）")

    embedded = extract_embedded_paths(notebook)
    print(f"内嵌源码文件数 = {len(embedded)}（工程源码文件数 = {len(discover_source_files())}）")

    # ---- 完整性 + 一致性自检：任一问题都以非零退出码报错，避免"静默缺文件"---- #
    # 说明：这里只用 ASCII 标记，避免在 Windows GBK 控制台上触发 UnicodeEncodeError。
    problems = verify_notebook(path)
    if problems:
        print("\n[FAIL] Notebook 自检未通过：")
        for item in problems:
            print(f"    - {item}")
        raise SystemExit(1)
    print("[OK] 自检通过：完整性与逐字一致性均无误")
    print("  内嵌文件：")
    for rel in embedded:
        print(f"    - {rel}")
