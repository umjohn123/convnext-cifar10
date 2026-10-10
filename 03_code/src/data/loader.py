# -*- coding: utf-8 -*-
"""
数据集组装与 DataLoader 构建。

本模块负责把 :mod:`src.data.cifar10` / :mod:`src.data.synthetic` 提供的"数据底座"
组装成训练 / 验证 / 测试三个数据集与对应的 DataLoader。

数据划分协议（与既有结果严格一致，不可随意更改）
------------------------------------------------
* **训练池**：官方训练集 50000 张，用 ``torch.Generator().manual_seed(--seed)``
  的 ``torch.randperm(50000)`` 打乱；
* **验证集**：取该排列的**后 10%**（默认 5000 张），**不参与训练**；
* **测试集**：官方测试集 10000 张，**完全不参与训练与调参**；
* **快速验证子集**（可选 ``--train-subset-size`` / ``--test-subset-size``）：
  从同一排列中截取前 N 个，因此与全量划分**完全一致且可互相校验**。

由于索引语义直接决定已归档结果的可复现性，本模块的随机数使用方式与顺序
**必须保持原样**；`04_data/splits/*.pt` 中保存的索引可用于回归校验。

性能说明（实测优化）
--------------------
``torchvision.datasets.CIFAR10`` 每次构造都会对全部批次文件做 MD5 完整性校验，
并把 pickle 反序列化到内存。原实现为"带增强训练集 / 不带增强验证集 / 测试集"
各构造一个实例，导致：

* CIFAR-10 被实例化 **3 次** → 完整性校验被触发 **6 次**（约 186 MB × 6 的哈希）；
* 训练集 pickle（146 MB）被读入内存 **2 份**。

实测热路径耗时 4.5 s，其中几乎全部由重复 MD5 校验贡献。现改为：
**每个 split 只构造一个 ``transform=None`` 的底座**，再通过
:class:`TransformSubset` 叠加不同变换 —— 划分与索引语义不变，开销显著下降。
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Sequence, Tuple

from ..utils.logger import get_logger
from ..utils.seed import make_generator, seed_worker
from .cifar10 import (
    CIFAR10_CLASSES,
    build_cifar10_base,
    build_test_transform,
    build_train_transform,
    is_extracted,
)
from .synthetic import build_synthetic_dataset

#: 官方划分规模（用于判断是否为全量实验）
OFFICIAL_TRAIN_SIZE = 50000
OFFICIAL_TEST_SIZE = 10000

#: 识别 CIFAR-10 的数据集名称别名
_CIFAR10_ALIASES = {"cifar10", "cifar-10", "cifar"}
#: 识别合成数据的名称别名
_SYNTHETIC_ALIASES = {"synthetic", "fake", "dummy"}


# --------------------------------------------------------------------------- #
# 变换包装器
# --------------------------------------------------------------------------- #
class TransformSubset(Dataset):
    """在**同一个底座数据集**上，用指定索引与指定变换取样本。

    为什么需要它
    ------------
    训练集需要"带增强"与"不带增强（验证用）"两个版本，而
    ``torchvision.datasets.CIFAR10`` 的 ``transform`` 在构造时就固定了。
    若各构造一个实例，会重复触发 MD5 完整性校验与 pickle 反序列化。
    本包装器让两种变换共享同一个 ``transform=None`` 的底座，从而消除该开销，
    同时**不改变任何索引语义**（取出的是同一个底座的第 ``indices[i]`` 个样本）。

    参数
    ----
    base : Dataset
        底座数据集，其 ``__getitem__`` 返回 ``(PIL.Image, int)``。
    indices : Sequence[int]
        底座中的样本下标（顺序即子集顺序）。
    transform : callable | None
        作用于 PIL 图像的变换；None 表示不变换。
    """

    def __init__(self, base: Dataset, indices: Sequence[int], transform=None) -> None:
        self.base = base
        self.indices: List[int] = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[object, int]:
        image, target = self.base[self.indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


# --------------------------------------------------------------------------- #
# 底座数据集
# --------------------------------------------------------------------------- #
def build_base_datasets(cfg, logger=None) -> Tuple[Dataset, Dataset, str]:
    """按 ``cfg.data.dataset`` 构造 ``(train_base, test_base, dataset_name)``。

    * ``dataset="CIFAR10"``：真实数据集；仅在缓存缺失时才允许下载，
      避免每次都走"下载分支"而多触发一轮完整性校验；
    * ``dataset="synthetic"``：本地生成的合成数据，**零下载**，
      用于网络受限环境下的流程冒烟测试。

    返回的底座均 **不带变换**（``transform=None``），由 :class:`TransformSubset` 叠加。
    """
    name = str(getattr(cfg.data, "dataset", "CIFAR10")).strip()
    key = name.lower()

    if key in _SYNTHETIC_ALIASES:
        train_base = build_synthetic_dataset(cfg, train=True, logger=logger)
        test_base = build_synthetic_dataset(cfg, train=False, logger=logger)
        return train_base, test_base, "synthetic"

    if key in _CIFAR10_ALIASES:
        root = cfg.data.data_root
        cached = is_extracted(root)
        if logger is not None:
            logger.info(f"[data] CIFAR-10 缓存状态：{'已就绪（跳过下载与重复校验）' if cached else '缺失，将自动下载'}")
        # 只在缓存缺失时要求下载；测试集与训练集来自同一个归档，因此无需重复下载
        train_base = build_cifar10_base(cfg, train=True, download=not cached, logger=logger)
        test_base = build_cifar10_base(cfg, train=False, download=False, logger=logger)
        return train_base, test_base, "CIFAR-10"

    raise ValueError(
        f"未知的 dataset='{name}'，可选：'CIFAR10'（真实数据）或 'synthetic'（零下载合成数据）"
    )


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
def build_datasets(cfg, logger=None) -> Tuple[Dataset, Dataset, Dataset, List[str]]:
    """构造并返回 ``(train_set, val_set, test_set, class_names)``。

    划分逻辑与历史结果严格一致（见模块文档字符串）。启用"快速验证子集"时，
    日志会打印**显式警告**，提醒其指标不可与全量结果比较。
    """
    log = logger if logger is not None else get_logger()
    train_base, test_base, dataset_name = build_base_datasets(cfg, logger=logger)

    train_transform = build_train_transform(
        mean=cfg.data.mean, std=cfg.data.std,
        rand_augment=cfg.data.rand_augment, random_erase=cfg.data.random_erase,
    )
    eval_transform = build_test_transform(mean=cfg.data.mean, std=cfg.data.std)

    # ---------------- 训练池的固定种子排列 ---------------- #
    total = len(train_base)
    generator = torch.Generator().manual_seed(int(cfg.train.seed))
    perm = torch.randperm(total, generator=generator).tolist()

    # ---------------- 可选的"快速验证子集" ---------------- #
    pool_indices = perm
    subset_size = getattr(cfg.data, "train_subset_size", None)
    used_train_subset = False
    if subset_size is not None and 0 < int(subset_size) < total:
        pool_indices = perm[: int(subset_size)]
        used_train_subset = True

    pool_size = len(pool_indices)
    num_val = int(pool_size * float(cfg.data.val_ratio))
    num_train = pool_size - num_val

    train_indices = pool_indices[:num_train]
    val_indices = pool_indices[num_train:]

    # 训练与验证共享同一个底座，只是变换不同（这是本次性能优化的核心）
    train_set = TransformSubset(train_base, train_indices, train_transform)
    val_set = TransformSubset(train_base, val_indices, eval_transform)

    # ---------------- 测试集（可选子集） ---------------- #
    test_indices = list(range(len(test_base)))
    used_test_subset = False
    test_size = getattr(cfg.data, "test_subset_size", None)
    if test_size is not None and 0 < int(test_size) < len(test_base):
        # 使用与训练集不同的种子偏移，避免两个抽样序列相互关联
        test_generator = torch.Generator().manual_seed(int(cfg.train.seed) + 1)
        shuffled = torch.randperm(len(test_base), generator=test_generator).tolist()
        test_indices = shuffled[: int(test_size)]
        used_test_subset = True

    test_set = TransformSubset(test_base, test_indices, eval_transform)

    if used_train_subset or used_test_subset:
        log.warning(
            "[data] 已启用『快速验证子集』模式："
            + (f"训练池 {pool_size} / 原 {total} 张" if used_train_subset else "")
            + ("，" if used_train_subset and used_test_subset else "")
            + (f"测试集 {len(test_set)} / 原 {len(test_base)} 张" if used_test_subset else "")
            + "。该模式下的指标**不能**与全量数据结果直接比较，仅用于快速验证流程。"
        )
    if dataset_name == "synthetic":
        log.warning("[data] 当前使用合成数据，全部指标仅供流程验证，不具备任何科学意义。")

    class_names = list(getattr(train_base, "classes", CIFAR10_CLASSES))
    return train_set, val_set, test_set, class_names


# --------------------------------------------------------------------------- #
# DataLoader
# --------------------------------------------------------------------------- #
def build_dataloaders(
    cfg,
    batch_size: int,
    eval_batch_size: Optional[int] = None,
    logger=None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """构建训练 / 验证 / 测试三个 DataLoader。

    参数
    ----
    cfg : Config
        全局配置对象。
    batch_size : int
        训练 batch size（建议传入显存自适应探测后的结果）。
    eval_batch_size : int | None
        验证/测试 batch size；为 None 时取 ``2 * batch_size``（推理无梯度，可更大）。
    logger : Logger | None
        日志器，用于打印数据集规模与划分信息。

    返回
    ----
    (train_loader, val_loader, test_loader, class_names)
    """
    log = logger if logger is not None else get_logger()
    train_set, val_set, test_set, class_names = build_datasets(cfg, logger=logger)

    if eval_batch_size is None:
        eval_batch_size = max(1, int(batch_size) * 2)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    use_pin_memory = bool(cfg.data.pin_memory) and device_type == "cuda"
    num_workers = int(cfg.data.num_workers)

    # 多进程相关设置：num_workers=0 时必须关闭 persistent_workers
    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=int(batch_size),
        shuffle=True,                                     # 训练集每个 epoch 重新打乱
        drop_last=False,
        generator=make_generator(int(cfg.train.seed)),    # 固定 shuffle 顺序（可复现）
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(eval_batch_size),
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(eval_batch_size),
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    log.info(
        f"[data] {class_names and '数据集'} 划分完成 | 训练 {len(train_set)} 张 | "
        f"验证 {len(val_set)} 张 | 测试 {len(test_set)} 张 | 类别数 {len(class_names)}"
    )
    log.info(
        f"[data] DataLoader | train batch={batch_size} | eval batch={eval_batch_size} | "
        f"num_workers={num_workers} | pin_memory={use_pin_memory}"
    )
    return train_loader, val_loader, test_loader, class_names
