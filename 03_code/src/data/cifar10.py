# -*- coding: utf-8 -*-
"""
CIFAR-10 常量、预处理变换与"数据底座"构造。

本模块只负责三件事：

1. 定义 CIFAR-10 的类别名与归一化统计量（:data:`CIFAR10_CLASSES` 等）；
2. 定义训练 / 评估两套**确定性可控**的变换流水线；
3. 构造**不带变换**的数据底座（:func:`build_cifar10_base`）。

数据划分、子集抽取与 DataLoader 组装见 :mod:`src.data.loader`。

为什么把"底座"与"变换"解耦
---------------------------
``torchvision.datasets.CIFAR10`` 在 ``__init__`` 时会把 ``transform`` 固定下来，
并且**每次构造**都会执行两类代价不低的操作：

* **完整性校验**：对全部 6 个批次文件（共约 186 MB）逐个做 MD5；
* **pickle 反序列化**：训练集约 146 MB 数组读入内存。

训练集需要"带增强"和"不带增强"两个版本。若各构造一个 ``CIFAR10`` 实例，
上述两项开销都会翻倍（实测：热路径 4.5 s 里几乎全部由重复 MD5 校验贡献）。

因此本工程改为：**每个 split 只构造一个底座**（``transform=None``），
再用 :class:`src.data.loader.TransformSubset` 叠加不同变换，
既保证划分与索引语义**完全不变**，又消除重复校验与重复加载。
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

from torch.utils.data import Dataset
from torchvision import datasets, transforms

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-10 训练集在 [0, 1] 尺度下统计得到的通道均值与标准差
CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

#: CIFAR-10 图像尺寸
IMAGE_SIZE: int = 32

#: torchvision 解压后生成的目录名（用于判断数据是否已就绪）
BASE_FOLDER: str = "cifar-10-batches-py"

#: 官方压缩包文件名与 MD5（用于下载加速脚本与文档引用）
ARCHIVE_NAME: str = "cifar-10-python.tar.gz"
ARCHIVE_MD5: str = "c58f30108f718f92721af3b95e74349a"
ARCHIVE_SIZE: int = 170498071


# --------------------------------------------------------------------------- #
# 变换
# --------------------------------------------------------------------------- #
def build_train_transform(
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD,
    rand_augment: bool = False,
    random_erase: float = 0.25,
) -> transforms.Compose:
    """构建训练集的预处理与数据增强流水线。

    参数
    ----
    mean, std : Sequence[float]
        归一化使用的通道均值与标准差。
    rand_augment : bool
        是否叠加 RandAugment（更激进的自动增强，小数据集上可能过强，默认关闭）。
    random_erase : float
        RandomErasing 的触发概率；<=0 表示关闭该增强。
    """
    ops: List[object] = [
        # CIFAR 图像的填充方式使用 reflect 效果通常优于默认的常量填充
        transforms.RandomCrop(IMAGE_SIZE, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if rand_augment:
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops += [
        transforms.ToTensor(),                      # HWC uint8 -> CHW float，并缩放到 [0, 1]
        transforms.Normalize(mean, std),            # 标准化到零均值、单位方差
    ]
    if random_erase is not None and random_erase > 0:
        # RandomErasing 必须在 Normalize 之后（它作用在 Tensor 上，单位已归一化）
        ops.append(transforms.RandomErasing(p=float(random_erase), scale=(0.02, 0.2), value=0.0))
    return transforms.Compose(ops)


def build_test_transform(
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD,
) -> transforms.Compose:
    """构建验证/测试集的确定性预处理（仅 ToTensor + Normalize，不做增强）。"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# --------------------------------------------------------------------------- #
# 数据底座
# --------------------------------------------------------------------------- #
def is_extracted(root: str) -> bool:
    """判断 CIFAR-10 是否已解压就绪（不触发任何校验或下载）。"""
    return os.path.isfile(os.path.join(root, BASE_FOLDER, "test_batch"))


def build_cifar10_base(
    cfg,
    train: bool,
    download: bool = False,
    logger=None,
) -> Dataset:
    """构造一个**不带变换**的 CIFAR-10 底座数据集。

    参数
    ----
    cfg : Config
        全局配置对象（读取 ``data.data_root``）。
    train : bool
        True 取官方训练集（50000 张），False 取官方测试集（10000 张）。
    download : bool
        数据缺失时是否允许下载。调用方应先用 :func:`is_extracted` 判断缓存情况，
        **只有在确实缺失时才传 True**，以避免每次都触发下载分支中的额外校验。
    logger : Logger | None
        日志器，用于打印数据目录与是否复用缓存。

    返回
    ----
    Dataset
        返回的样本为 ``(PIL.Image, int)``，变换由调用方（``TransformSubset``）叠加。
    """
    root = cfg.data.data_root
    if download:
        os.makedirs(root, exist_ok=True)

    if logger is not None:
        logger.info(
            f"[data] 加载 CIFAR-10 {'训练集' if train else '测试集'}底座"
            f"（transform=None，目录 {os.path.abspath(root)}）"
        )

    return datasets.CIFAR10(root=root, train=train, download=download, transform=None)
