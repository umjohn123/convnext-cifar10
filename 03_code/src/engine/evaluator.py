# -*- coding: utf-8 -*-
"""
测试 / 验证评估模块。

包含四类功能：
1. :func:`topk_accuracy` —— Top-k 准确率计算（k=1,5）；
2. :func:`evaluate`      —— 在给定 DataLoader 上计算平均损失与 Top-1/Top-5 准确率；
3. :func:`collect_predictions` —— 收集全部预测结果与标签（用于混淆矩阵、误差分析）；
4. :func:`analyze_predictions` —— 逐类准确率、混淆矩阵、分类报告，并落盘为 JSON。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.amp import autocast_context
from ..utils.misc import AverageMeter, save_json
from ..utils.tta import tta_average_probs


# --------------------------------------------------------------------------- #
# 1. Top-k 准确率
# --------------------------------------------------------------------------- #
@torch.no_grad()
def topk_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    topk: Sequence[int] = (1, 5),
) -> List[float]:
    """计算一个 batch 的 Top-k 准确率（百分比，0~100）。

    参数
    ----
    outputs : Tensor, shape (N, C)
        模型输出的 logits（未经过 softmax）。
    targets : Tensor, shape (N,)
        真实标签。
    topk : Sequence[int]
        需要计算的 k 列表。

    返回
    ----
    List[float]
        与 ``topk`` 一一对应的准确率（百分数）。

    实现细节
    --------
    借助 ``topk`` + ``eq`` 的广播比较，一次性算出所有 k，避免多次排序。
    当类别数小于 max(k) 时（如 CIFAR-10 只有 10 类），会自动把 k 收敛到有效范围。
    """
    num_classes = outputs.size(1)
    effective_k = sorted({min(int(k), num_classes) for k in topk})
    max_k = max(effective_k)

    batch_size = targets.size(0)
    _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)   # (N, max_k)
    pred = pred.t()                                                   # (max_k, N)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))            # (max_k, N) bool

    results: Dict[int, float] = {}
    for k in effective_k:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0)
        results[k] = float(correct_k.mul_(100.0 / max(1, batch_size)).item())
    return [results[min(int(k), num_classes)] for k in topk]


# --------------------------------------------------------------------------- #
# 2. 通用评估
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool = True,
    topk: Sequence[int] = (1, 5),
    max_batches: Optional[int] = None,
    tta: bool = False,
    tta_hflip: bool = True,
) -> Dict[str, float]:
    """在给定 DataLoader 上评估模型。

    参数
    ----
    model : nn.Module
        待评估模型（函数内部会切换到 ``eval()`` 模式）。
    loader : DataLoader
        验证集或测试集的数据加载器。
    criterion : nn.Module
        损失函数（一般与训练一致，带标签平滑）。
    device : torch.device
        计算设备。
    amp : bool
        是否启用混合精度推理（仅 CUDA 生效，可显著加速）。
    topk : Sequence[int]
        需要统计的 Top-k 指标。
    max_batches : int | None
        最多评估多少个 batch（用于快速冒烟测试）；None 表示全量评估。
    tta : bool
        是否启用**测试时增强**（原图 + 水平翻转两路，平均 softmax 概率后再取 Top-k）。
        推理耗时约翻倍，通常可带来 +0.2 ~ +0.5 个百分点的 Top-1。
    tta_hflip : bool
        TTA 是否包含水平翻转视角（``tta=True`` 时生效）。

   返回
    ----
    dict
        ``{"loss": float, "top1": float, "top5": float, "num_samples": int}``，
        其中准确率为百分数。

    说明
    ----
    为避免"损失与准确率口径不一致"，``loss`` 始终由**不做 TTA 的原始前向**计算
    （与训练期验证、非 TTA 评估严格可比）；只有 Top-k 准确率使用 TTA 平均概率。
    返回值中另含 ``tta`` 字段用于标记本次评估是否启用了 TTA。
    """
    model.eval()
    loss_meter = AverageMeter("loss")
    topk_meters = {int(k): AverageMeter(f"top{k}") for k in topk}

    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()

        with autocast_context(device, enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)
            if tta:
                # 传入已算好的原图 logits，避免重复前向
                scores = tta_average_probs(model, images, base_logits=logits, use_hflip=tta_hflip)
            else:
                scores = logits

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        accs = topk_accuracy(scores.float(), targets, topk=topk)
        for k, acc in zip(topk, accs):
            topk_meters[min(int(k), scores.size(1))].update(acc, batch_size)

    metrics: Dict[str, float] = {
        "loss": loss_meter.avg,
        "num_samples": loss_meter.count,
        "tta": 1.0 if tta else 0.0,
    }
    for k, meter in topk_meters.items():
        metrics[f"top{k}"] = meter.avg
    return metrics


# --------------------------------------------------------------------------- #
# 3. 收集预测结果
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
    max_batches: Optional[int] = None,
    tta: bool = False,
    tta_hflip: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """遍历数据集并收集所有预测标签与真实标签。

    参数
    ----
    tta : bool
        是否启用测试时增强（与原图 + 水平翻转的平均概率保持一致），
        需与 :func:`evaluate` 的 ``tta`` 参数取相同值，否则混淆矩阵与总体准确率口径不一致。
    tta_hflip : bool
        TTA 是否包含水平翻转视角。

    返回
    ----
    (y_true, y_pred) : tuple[np.ndarray, np.ndarray]
        两个形状均为 ``(N,)`` 的整数数组。
    """
    model.eval()
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        with autocast_context(device, enabled=amp):
            logits = model(images)
            if tta:
                scores = tta_average_probs(model, images, base_logits=logits, use_hflip=tta_hflip)
            else:
                scores = logits
        preds = scores.float().argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())

    if not all_preds:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(all_targets), np.concatenate(all_preds)


# --------------------------------------------------------------------------- #
# 4. 预测结果分析
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """计算混淆矩阵（不依赖 scikit-learn）。

    返回
    ----
    np.ndarray, shape (C, C)
        ``cm[i, j]`` = 真实类别 i 被预测为 j 的样本数。
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """由混淆矩阵计算逐类准确率（即召回率）。"""
    row_sum = cm.sum(axis=1).astype(np.float64)
    correct = np.diag(cm).astype(np.float64)
    return np.divide(correct, np.maximum(row_sum, 1e-12))


def analyze_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    logger=None,
    save_path: Optional[str] = None,
) -> Dict[str, object]:
    """汇总预测结果：总体准确率、逐类准确率、混淆矩阵。

    参数
    ----
    y_true, y_pred : np.ndarray
        :func:`collect_predictions` 的输出。
    class_names : Sequence[str]
        类别名称。
    logger : Logger | None
        日志器，若提供则打印每类准确率。
    save_path : str | None
        结果 JSON 的保存路径。

    返回
    ----
    dict
        包含 ``accuracy / per_class_accuracy / confusion_matrix / macro_accuracy`` 等字段。
    """
    num_classes = len(class_names)
    total = int(len(y_true))
    correct = int((y_true == y_pred).sum())
    overall_acc = 100.0 * correct / max(1, total)

    cm = confusion_matrix(y_true, y_pred, num_classes)
    per_class = per_class_accuracy(cm)

    result: Dict[str, object] = {
        "num_samples": total,
        "num_correct": correct,
        "accuracy": overall_acc,
        "macro_accuracy": float(np.mean(per_class) * 100.0),
        "per_class_accuracy": {name: float(per_class[i]) for i, name in enumerate(class_names)},
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }

    if logger is not None:
        logger.info("逐类准确率（测试集）：")
        for i, name in enumerate(class_names):
            logger.info(f"    {name:<11s}: {per_class[i] * 100.0:6.2f}%  "
                        f"(支撑样本 {int(cm[i].sum())} 张)")
        logger.info(f"宏平均（各类准确率均值）：{np.mean(per_class) * 100.0:.2f}%")

    if save_path:
        save_json(result, save_path)

    return result
