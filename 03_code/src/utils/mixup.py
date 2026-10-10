# -*- coding: utf-8 -*-
"""
Mixup 与 CutMix 数据增强。

ConvNeXt 原论文的训练配方中使用了 Mixup / CutMix 来提升精度与鲁棒性：

* **Mixup**（Zhang et al., 2018）：两张图像按 Beta 分布采样得到的比例做线性插值，
  标签同样按该比例混合 —— ``x = λ·x_i + (1-λ)·x_j``。
* **CutMix**（Yun et al., 2019）：把一张图像的矩形区域直接裁剪并粘贴到另一张图像上，
  λ 由矩形面积占比决定 —— 相比 Mixup，它保留了局部真实的像素统计。

两者的共同点是：一个 batch 内构造出"软标签"，
因此损失函数需要用 :func:`mixup_criterion` 做加权求和。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def rand_bbox(size: Tuple[int, int, int, int], lam: float) -> Tuple[int, int, int, int]:
    """按 λ 计算 CutMix 的矩形框坐标 (x1, y1, x2, y2)。

    参数
    ----
    size : tuple
        输入张量形状 ``(N, C, H, W)``。
    lam : float
        从 Beta 分布采样的混合系数，矩形面积占比约为 ``1 - lam``。
    """
    _, _, height, width = size
    ratio = float(np.sqrt(1.0 - lam))          # 裁剪边长比例
    cut_w = int(width * ratio)
    cut_h = int(height * ratio)

    # 矩形中心在图像内随机取点
    cx = int(np.random.randint(width))
    cy = int(np.random.randint(height))

    x1 = int(np.clip(cx - cut_w // 2, 0, width))
    y1 = int(np.clip(cy - cut_h // 2, 0, height))
    x2 = int(np.clip(cx + cut_w // 2, 0, width))
    y2 = int(np.clip(cy + cut_h // 2, 0, height))
    return x1, y1, x2, y2


def mixup(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """对 batch 内样本做 Mixup。

    返回
    ----
    mixed_images : Tensor
        混合后的图像。
    targets_a, targets_b : Tensor
        混合前的两组标签。
    lam : float
        targets_a 对应的权重（targets_b 的权重为 ``1 - lam``）。
    """
    lam = float(np.random.beta(alpha, alpha))
    # 生成 batch 内的随机置换索引
    index = torch.randperm(images.size(0), device=images.device)
    mixed = images * lam + images[index] * (1.0 - lam)
    return mixed, targets, targets[index], lam


def cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """对 batch 内样本做 CutMix。

    注意：λ 需要用"实际粘贴面积占比"重新计算，而不是直接用 Beta 采样值。
    """
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(images.size(0), device=images.device)

    # 先把另一张图备份出来，再做原地粘贴，避免"读到自己刚写入的内容"
    shuffled = images[index].clone()
    x1, y1, x2, y2 = rand_bbox(images.size(), lam)
    images[:, :, y1:y2, x1:x2] = shuffled[:, :, y1:y2, x1:x2]

    # 根据真实粘贴面积重算 λ
    _, _, height, width = images.size()
    area = max(0, (x2 - x1)) * max(0, (y2 - y1))
    lam = 1.0 - area / float(height * width)
    return images, targets, targets[index], lam


def mixup_criterion(
    criterion: nn.Module,
    predictions: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Mixup / CutMix 的加权交叉熵损失。

    ``criterion`` 一般带标签平滑（label smoothing），两个分支都会享受到平滑，
    这也是 ConvNeXt 官方训练脚本的做法。
    """
    return lam * criterion(predictions, targets_a) + (1.0 - lam) * criterion(predictions, targets_b)


def apply_mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    use_mixup: bool,
    prob: float,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
    """统一入口：以 ``prob`` 概率随机选择 Mixup 或 CutMix，否则保持原样。

    返回
    ----
    images, targets_a, targets_b, lam, applied
        ``applied=False`` 时 ``targets_a == targets_b == targets`` 且 ``lam=1.0``，
        可直接套用 :func:`mixup_criterion` 而无需分支判断。
    """
    if (not use_mixup) or (prob <= 0.0) or (np.random.rand() >= prob):
        return images, targets, targets, 1.0, False

    # 50% 概率二选一
    if np.random.rand() < 0.5:
        images, targets_a, targets_b, lam = mixup(images, targets, alpha=mixup_alpha)
    else:
        images, targets_a, targets_b, lam = cutmix(images, targets, alpha=cutmix_alpha)
    return images, targets_a, targets_b, lam, True
