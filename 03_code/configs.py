# -*- coding: utf-8 -*-
"""
全局配置模块（ConvNeXt + CIFAR-10）。

本文件集中管理**数据 / 模型 / 训练**三组超参数，避免超参数散落在代码各处。
所有字段都可以通过命令行参数（见 ``train.py`` 与 ``evaluate.py``）覆盖。

设计说明
--------
* 使用 ``dataclass`` 组织配置，便于 `asdict()` 序列化为 JSON 存档（可复现性证据）。
* 训练相关的默认值参考 ConvNeXt 原论文的训练配方：
  AdamW + 余弦退火 + 热身（warmup）+ 标签平滑（label smoothing）
  + Mixup / CutMix + 权重衰减 0.05（不对权重为 1 维的参数施加）。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

# 模型预设名称列表（真实定义见 src/models/convnext.py）
CONVNEXT_PRESET_NAMES: Tuple[str, ...] = ("atto", "femto", "pico", "nano", "tiny")


# --------------------------------------------------------------------------- #
# 1. 数据配置
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """CIFAR-10 数据加载与预处理配置。"""

    # 数据集类型：
    #   "CIFAR10"   —— 真实数据集（首次运行需下载约 170 MB）
    #   "synthetic" —— 本地生成的合成数据，**零下载**，用于网络受限时的流程冒烟测试
    dataset: str = "CIFAR10"
    data_root: str = "./data"          # 数据集下载 / 缓存目录
    num_classes: int = 10              # CIFAR-10 有 10 个类别
    val_ratio: float = 0.1             # 从 50000 张训练集中划出 10% 作为验证集
    # 合成数据规模（仅 dataset="synthetic" 时生效）
    synthetic_train_size: int = 4096
    synthetic_test_size: int = 2048
    # ---- 快速验证子集（CPU 上做 MVP/冒烟实验用；None 表示使用全量数据）---- #
    # 二者均按固定随机种子抽样，且会写入结果文件，保证可追溯、可复现。
    train_subset_size: Optional[int] = None   # 从 50000 张训练池中抽取的样本数
    test_subset_size: Optional[int] = None    # 从 10000 张测试集中抽取的样本数
    num_workers: int = 2               # Colab 免费版一般是 2 核 CPU
    pin_memory: bool = True            # GPU 训练时开启可加速 Host->Device 拷贝
    random_erase: float = 0.25         # RandomErasing 的应用概率；<=0 表示关闭
    rand_augment: bool = False         # 是否叠加 RandAugment（默认关闭，避免过强增强）
    # CIFAR-10 训练集统计得到的通道均值与标准差（在 ToTensor() 缩放到 [0,1] 之后）
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


# --------------------------------------------------------------------------- #
# 2. 模型配置
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """ConvNeXt 结构配置。"""

    name: str = "tiny"                 # atto / femto / pico / nano / tiny
    # 若为 None，则使用 name 对应的预设；也可显式指定以做消融实验
    depths: Optional[List[int]] = None            # 每个阶段的 Block 数量
    dims: Optional[List[int]] = None              # 每个阶段的通道数
    drop_path_rate: float = 0.1                   # Stochastic Depth 的最大丢弃率
    layer_scale_init_value: float = 1e-6          # Layer Scale 初始值
    stem_mode: str = "conv"                       # "conv"(3x3, s1，适配 32x32) | "patchify"(4x4, s4)
    head_init_scale: float = 1.0                  # 分类头初始权重的缩放系数


# --------------------------------------------------------------------------- #
# 3. 训练配置
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """训练与评估超参数。"""

    epochs: int = 60                   # 训练总轮数
    batch_size: int = 128              # **期望**的 batch size（显存不足时自动下调）
    auto_batch_size: bool = True       # 是否启用"显存自适应 batch size"
    max_batch_size: int = 512          # 自动探测时允许的最大 batch size
    min_batch_size: int = 8            # 自动探测时允许的最小 batch size
    lr: float = 1e-3                   # 基础学习率（AdamW）
    min_lr: float = 1e-5               # 余弦退火的下界学习率
    weight_decay: float = 0.05         # AdamW 权重衰减
    warmup_epochs: int = 5             # 线性热身轮数
    label_smoothing: float = 0.1       # 标签平滑系数
    grad_clip: float = 1.0             # 梯度裁剪阈值；<=0 表示关闭
    use_mixup: bool = True             # 是否启用 Mixup / CutMix
    mixup_prob: float = 0.5            # 每个 batch 应用 Mixup/CutMix 的概率
    mixup_alpha: float = 0.8           # Mixup 的 Beta 分布参数
    cutmix_alpha: float = 1.0          # CutMix 的 Beta 分布参数
    amp: bool = True                   # 是否在 GPU 上启用混合精度训练
    # AMP 安全网：fp16 的 GradScaler 动态 scale 一旦崩塌（典型值 <1），
    # 说明梯度已在下溢、部分参数停止更新；开启本项后会自动关闭 AMP 并以 fp32 续训。
    # 详见 src/utils/amp.py 与 src/engine/trainer.py::_track_amp_scale
    amp_fallback: bool = True
    tta: bool = False                  # 最终测试评估是否启用测试时增强（原图 + 水平翻转）
    tta_hflip: bool = True             # TTA 是否包含水平翻转视角（tta=True 时生效）
    seed: int = 42                     # 全局随机种子
    deterministic: bool = False        # 是否强制 cuDNN 确定性（会略微降低速度）
    plot_every: int = 10               # 每多少轮刷新一次曲线图；<=0 表示只在结束时绘制
    # 每个 epoch 最多训练多少个 step；None 表示不限制。
    # 用途：让"快速冒烟测试"的耗时**只取决于 step 数**，与数据集规模、硬件性能解耦 ——
    # 在只有 2 vCPU 的 Colab CPU 运行时上，即使很小的数据集，完整跑完一个 epoch 也可能超出预期。
    max_steps_per_epoch: Optional[int] = None
    early_stop_patience: int = 0       # 早停耐心值（以轮为单位）；0 表示不启用
    resume: Optional[str] = None       # 断点续训的 checkpoint 路径


# --------------------------------------------------------------------------- #
# 4. 顶层配置
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """完整配置对象，聚合数据、模型与训练配置，并派生输出目录。"""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "./outputs"      # 所有产物（模型/日志/图表）的根目录
    device: str = "auto"               # auto / cuda / mps / cpu
    exp_name: str = "convnext_cifar10" # 实验名称，写入日志与指标文件
    # 实验标签：设置后，全部产物写入 ``<output_dir>/<run_tag>/``，
    # 从而让"全量训练 / 对比实验 / 消融实验 / 多种子重复"各自独立成目录、
    # 互不覆盖、互不交叉引用。为 None 时保持旧行为（直接写入 output_dir）。
    run_tag: Optional[str] = None

    # ---------------- 派生路径 ---------------- #
    # 输出目录按"结果三分类"组织，与作业 05_results 的结构一一对应：
    #   figures(图表) / logs(日志) / models(模型) / metrics(指标)
    @property
    def run_root(self) -> str:
        """本次实验的产物根目录：``<output_dir>/<run_tag>``（未设置 run_tag 时即 output_dir）。"""
        if self.run_tag:
            return os.path.join(self.output_dir, str(self.run_tag))
        return self.output_dir

    @property
    def fig_dir(self) -> str:
        """图表目录：训练曲线、混淆矩阵、逐类准确率、数据样例。"""
        return os.path.join(self.run_root, "figures")

    @property
    def log_dir(self) -> str:
        """日志目录：训练/评估日志与逐轮训练历史。"""
        return os.path.join(self.run_root, "logs")

    @property
    def ckpt_dir(self) -> str:
        """模型目录：最优与最后一轮的检查点文件。"""
        return os.path.join(self.run_root, "models")

    @property
    def result_dir(self) -> str:
        """指标目录：超参数、测试指标、逐类指标、指标汇总表等结构化结果。"""
        return os.path.join(self.run_root, "metrics")

    def make_dirs(self) -> None:
        """创建所有输出目录（幂等）。"""
        for d in (self.output_dir, self.run_root, self.ckpt_dir, self.log_dir,
                  self.fig_dir, self.result_dir):
            os.makedirs(d, exist_ok=True)

    # ---------------- 序列化 ---------------- #
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """由字典重建 :class:`Config`（用于从检查点中恢复训练时的完整配置）。

        只保留与 dataclass 字段同名的键，多余键会被忽略，
        因此可以安全地读取旧版本保存的检查点。
        """

        def _filtered(dc_type, payload: dict) -> dict:
            fields = set(dc_type.__dataclass_fields__.keys())
            return {k: v for k, v in dict(payload or {}).items() if k in fields}

        data = data or {}
        return cls(
            data=DataConfig(**_filtered(DataConfig, data.get("data", {}))),
            model=ModelConfig(**_filtered(ModelConfig, data.get("model", {}))),
            train=TrainConfig(**_filtered(TrainConfig, data.get("train", {}))),
            output_dir=data.get("output_dir", cls.output_dir),
            device=data.get("device", cls.device),
            exp_name=data.get("exp_name", cls.exp_name),
            run_tag=data.get("run_tag", None),
        )

    def copy(self) -> "Config":
        """返回配置的深拷贝（避免在覆盖参数时污染原始对象）。"""
        return Config.from_dict(self.to_dict())


# --------------------------------------------------------------------------- #
# 5. 命令行参数解析
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None, base: Optional[Config] = None) -> Config:
    """解析命令行参数并返回覆盖后的 :class:`Config`。

    参数
    ----
    argv : list[str] | None
        参数列表；为 None 时读取 ``sys.argv[1:]``。
    base : Config | None
        基准配置。**只覆盖用户显式传入的参数**（所有选项的默认值均为 None），
        因此可以先用检查点中保存的配置作为基准，再由命令行做局部覆盖，
        从而保证评估时模型结构与训练时完全一致。

    返回
    ----
    Config
        合并命令行覆盖项后的配置对象。
    """
    parser = argparse.ArgumentParser(
        description="ConvNeXt on CIFAR-10 (PyTorch, Colab-friendly)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 数据
    parser.add_argument("--data-root", type=str, default=None, help="数据集缓存目录")
    parser.add_argument("--dataset", type=str, default=None, choices=["CIFAR10", "synthetic"],
                        help="数据集类型：CIFAR10（真实，需下载）或 synthetic（本地合成，零下载）")
    parser.add_argument("--synthetic-train-size", type=int, default=None,
                        help="合成数据训练集规模（dataset=synthetic 时生效）")
    parser.add_argument("--synthetic-test-size", type=int, default=None,
                        help="合成数据测试集规模（dataset=synthetic 时生效）")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader 工作进程数")
    parser.add_argument("--val-ratio", type=float, default=None, help="验证集比例")
    parser.add_argument("--train-subset-size", type=int, default=None,
                        help="快速验证子集：从训练池中抽取的样本数（默认使用全量 50000）")
    parser.add_argument("--test-subset-size", type=int, default=None,
                        help="快速验证子集：从测试集中抽取的样本数（默认使用全量 10000）")
    # 模型
    parser.add_argument("--model", type=str, default=None, choices=list(CONVNEXT_PRESET_NAMES),
                        help="ConvNeXt 预设规模")
    parser.add_argument("--stem-mode", type=str, default=None, choices=["conv", "patchify"],
                        help="Stem 类型：conv=3x3 s1（适配 32x32），patchify=论文 4x4 s4")
    parser.add_argument("--drop-path", type=float, default=None, help="Stochastic Depth 丢弃率")
    # 训练
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="期望 batch size")
    parser.add_argument("--lr", type=float, default=None, help="基础学习率")
    parser.add_argument("--weight-decay", type=float, default=None, help="权重衰减")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="线性热身轮数")
    parser.add_argument("--label-smoothing", type=float, default=None, help="标签平滑系数")
    parser.add_argument("--plot-every", type=int, default=None, help="每多少轮刷新一次曲线图（0=仅结束时绘制）")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None,
                        help="每个 epoch 最多训练多少个 step（用于限时冒烟测试；默认不限制）")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="早停耐心值（0=不启用）")
    parser.add_argument("--seed", type=int, default=None, help="全局随机种子")
    parser.add_argument("--resume", type=str, default=None,
                        help="断点续训的 checkpoint 路径；填 'auto' 则自动从本次实验的 last.pt 续训")
    parser.add_argument("--output-dir", type=str, default=None, help="输出根目录")
    parser.add_argument("--exp-name", type=str, default=None, help="实验名称")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="实验标签：产物写入 <output-dir>/<run-tag>/，实现实验间目录隔离")
    parser.add_argument("--device", type=str, default=None, help="auto / cuda / mps / cpu")
    # 开关型参数
    parser.add_argument("--no-amp", action="store_true", help="关闭混合精度训练（fp16 下溢时的兜底做法）")
    parser.add_argument("--no-amp-fallback", action="store_true",
                        help="禁用 AMP 安全网：GradScaler scale 崩塌时不再自动降级为 fp32")
    parser.add_argument("--tta", action="store_true",
                        help="最终测试评估启用 TTA（原图 + 水平翻转，平均 softmax 概率）")
    parser.add_argument("--no-tta-hflip", action="store_true",
                        help="TTA 只保留原图视角（等价于不启用翻转分支）")
    parser.add_argument("--no-mixup", action="store_true", help="关闭 Mixup / CutMix")
    parser.add_argument("--no-auto-batch-size", action="store_true", help="关闭显存自适应 batch size")
    parser.add_argument("--deterministic", action="store_true", help="强制 cuDNN 确定性算法")

    args = parser.parse_args(argv)
    cfg = base.copy() if base is not None else Config()

    # ---- 数据 ----
    if args.data_root is not None:
        cfg.data.data_root = args.data_root
    if args.dataset is not None:
        cfg.data.dataset = args.dataset
    if args.synthetic_train_size is not None:
        cfg.data.synthetic_train_size = args.synthetic_train_size
    if args.synthetic_test_size is not None:
        cfg.data.synthetic_test_size = args.synthetic_test_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.val_ratio is not None:
        cfg.data.val_ratio = args.val_ratio
    if args.train_subset_size is not None:
        cfg.data.train_subset_size = args.train_subset_size
    if args.test_subset_size is not None:
        cfg.data.test_subset_size = args.test_subset_size

    # ---- 模型 ----
    if args.model is not None:
        cfg.model.name = args.model
    if args.stem_mode is not None:
        cfg.model.stem_mode = args.stem_mode
    if args.drop_path is not None:
        cfg.model.drop_path_rate = args.drop_path

    # ---- 训练 ----
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.weight_decay is not None:
        cfg.train.weight_decay = args.weight_decay
    if args.warmup_epochs is not None:
        cfg.train.warmup_epochs = args.warmup_epochs
    if args.label_smoothing is not None:
        cfg.train.label_smoothing = args.label_smoothing
    if args.plot_every is not None:
        cfg.train.plot_every = args.plot_every
    if args.max_steps_per_epoch is not None:
        cfg.train.max_steps_per_epoch = args.max_steps_per_epoch
    if args.early_stop_patience is not None:
        cfg.train.early_stop_patience = args.early_stop_patience
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.resume is not None:
        cfg.train.resume = args.resume
    if args.no_amp:
        cfg.train.amp = False
    if args.no_amp_fallback:
        cfg.train.amp_fallback = False
    if args.tta:
        cfg.train.tta = True
    if args.no_tta_hflip:
        cfg.train.tta_hflip = False
    if args.no_mixup:
        cfg.train.use_mixup = False
    if args.no_auto_batch_size:
        cfg.train.auto_batch_size = False
    if args.deterministic:
        cfg.train.deterministic = True

    # ---- 其他 ----
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.exp_name is not None:
        cfg.exp_name = args.exp_name
    if args.run_tag is not None:
        cfg.run_tag = args.run_tag
    if args.device is not None:
        cfg.device = args.device

    return cfg
