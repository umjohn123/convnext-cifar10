# -*- coding: utf-8 -*-
"""
测试时增强（Test-Time Augmentation, TTA）。

原理
----
对同一批图像构造多组**确定性**的视角，分别前向得到类别概率，再取算术平均，
用平均后的概率做预测。图像分类中性价比最高的是「原图 + 水平翻转」两路 TTA：
代价约等于一次额外前向（推理时间 ×2），通常可带来 **+0.2 ~ +0.5 个百分点**的 Top-1。

与训练时增强的区别
------------------
训练时的 ``RandomHorizontalFlip`` 是**随机采样**，用于扩充数据分布；
TTA 的翻转是**成对、确定性**的：只是把"翻转前 / 翻转后"两个视角的预测平均，
不引入任何随机性，因此结果**完全可复现**。

实现细节
--------
* 概率使用 ``softmax`` 后求平均（先归一化再平均，避免不同视角 logits 尺度差异的影响）；
* 若调用方已经算过原图 logits，可通过 ``base_logits`` 传入以**避免重复前向**；
* 水平翻转对应张量第 3 维（``N, C, H, W`` 中的 ``W``）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

#: 支持的 TTA 视角（当前实现：原图 + 水平翻转）
SUPPORTED_TTA_VIEWS = ("identity", "hflip")


def hflip(x: torch.Tensor) -> torch.Tensor:
    """按宽度维做水平翻转（``N, C, H, W`` 中翻转 ``W``）。"""
    return torch.flip(x, dims=[3])


@torch.no_grad()
def tta_average_probs(
    model: nn.Module,
    images: torch.Tensor,
    base_logits: Optional[torch.Tensor] = None,
    use_hflip: bool = True,
) -> torch.Tensor:
    """计算多视角 TTA 的平均类别概率。

    参数
    ----
    model : nn.Module
        模型（调用方需保证已处于 ``eval()`` 模式）。
    images : Tensor
        输入图像，形状 ``(N, C, H, W)``。
    base_logits : Tensor | None
        原图视角的 logits；为 None 时函数内部会自行前向一次。
    use_hflip : bool
        是否加入水平翻转视角。为 False 时退化为普通的单视角预测
        （此时返回 softmax 概率，与 ``tta`` 语义一致但无增益）。

    返回
    ----
    Tensor
        形状 ``(N, num_classes)`` 的平均概率，``.sum(dim=1) ≈ 1``。

    备注
    ----
    函数内部用 ``float32`` 做 softmax 与平均，避免半精度下的数值误差累积；
    调用方可放心在 ``autocast`` 上下文中使用。
    """
    if base_logits is None:
        base_logits = model(images)
    probs = torch.softmax(base_logits.float(), dim=1)

    if use_hflip:
        flip_logits = model(hflip(images))
        probs = probs + torch.softmax(flip_logits.float(), dim=1)
        probs = probs / float(len(SUPPORTED_TTA_VIEWS))

    return probs


@torch.no_grad()
def tta_predict(
    model: nn.Module,
    images: torch.Tensor,
    use_hflip: bool = True,
) -> torch.Tensor:
    """TTA 推理并返回预测类别索引，形状 ``(N,)``。"""
    probs = tta_average_probs(model, images, base_logits=None, use_hflip=use_hflip)
    return probs.argmax(dim=1)


def describe_tta(use_hflip: bool = True) -> str:
    """返回 TTA 配置的可读描述，用于写入日志与结果文件。"""
    views: Sequence[str] = ("identity", "hflip") if use_hflip else ("identity",)
    return f"TTA views = {list(views)} (平均 softmax 概率)"
