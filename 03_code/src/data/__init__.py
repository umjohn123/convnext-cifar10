# -*- coding: utf-8 -*-
"""数据模块：数据集构造、划分、预处理与 DataLoader。

结构
----
* :mod:`src.data.cifar10`   —— CIFAR-10 常量、变换与数据底座；
* :mod:`src.data.synthetic` —— 零下载合成数据（冒烟测试用）；
* :mod:`src.data.loader`    —— 划分协议、``TransformSubset`` 与 DataLoader 组装。
"""

from .cifar10 import (
    ARCHIVE_MD5,
    ARCHIVE_NAME,
    ARCHIVE_SIZE,
    BASE_FOLDER,
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    IMAGE_SIZE,
    build_cifar10_base,
    build_test_transform,
    build_train_transform,
    is_extracted,
)
from .loader import (
    OFFICIAL_TEST_SIZE,
    OFFICIAL_TRAIN_SIZE,
    TransformSubset,
    build_base_datasets,
    build_dataloaders,
    build_datasets,
)
from .synthetic import SyntheticCIFAR, build_synthetic_dataset

__all__ = [
    # 常量
    "ARCHIVE_MD5",
    "ARCHIVE_NAME",
    "ARCHIVE_SIZE",
    "BASE_FOLDER",
    "CIFAR10_CLASSES",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "IMAGE_SIZE",
    "OFFICIAL_TEST_SIZE",
    "OFFICIAL_TRAIN_SIZE",
    # 数据集
    "SyntheticCIFAR",
    "TransformSubset",
    "build_base_datasets",
    "build_cifar10_base",
    "build_dataloaders",
    "build_datasets",
    "build_synthetic_dataset",
    "build_test_transform",
    "build_train_transform",
    "is_extracted",
]
