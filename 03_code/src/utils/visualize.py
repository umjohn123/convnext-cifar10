# -*- coding: utf-8 -*-
"""
可视化工具：训练曲线（损失 / 准确率 / 学习率）、混淆矩阵、数据样本网格。

所有绘图函数都会把图片保存到 ``outputs/figures/``，
供报告（``06_report/``）与结果归档（``05_results/``）直接引用。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# 说明：此处**不强制**指定 matplotlib 后端，以便两种环境都能正常工作：
#   * Google Colab / Jupyter：自动使用 inline 后端，plt.show() 可内联显示图片；
#   * 无显示环境的服务器 / 后台脚本：自动回退到 Agg 后端，savefig 依然可以正常出图。


# 统一的配色
_COLOR_TRAIN = "#2E86C1"
_COLOR_VAL = "#E74C3C"
_COLOR_LR = "#27AE60"

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def _ensure_dir(path: str) -> None:
    """确保输出图片的父目录存在。"""
    if path:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)


def _finalize(fig, save_path: Optional[str], show: bool):
    """统一收尾：保存图片 ->（可选）显示 -> 释放图形对象，避免内存泄漏。"""
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# 1. 训练曲线
# --------------------------------------------------------------------------- #
def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "ConvNeXt on CIFAR-10",
    show: bool = False,
):
    """绘制训练/验证的**损失曲线**与**准确率曲线**（外加学习率曲线）。

    参数
    ----
    history : dict
        至少包含键 ``epoch, train_loss, train_acc, val_loss, val_acc, lr``。
        ``train_acc`` / ``val_acc`` 以 0~1 或 0~100 的百分比形式给出均可。
    save_path : str | None
        保存路径（PNG）。为 None 时不保存。
    title : str
        图标题。
    show : bool
        是否调用 ``plt.show()``（在 Colab 中可显示内联图像）。

    返回
    ----
    matplotlib.figure.Figure
    """
    epochs = list(range(1, len(history.get("epoch", history.get("train_loss", []))) + 1))
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])
    lr = history.get("lr", [])

    # 若准确率是 0~1 的小数，统一换算成百分比便于阅读
    def _to_percent(seq: Sequence[float]) -> List[float]:
        if not seq:
            return []
        return [v * 100.0 for v in seq] if max(seq) <= 1.0 else list(seq)

    train_acc_pct, val_acc_pct = _to_percent(train_acc), _to_percent(val_acc)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # ---- (a) 损失曲线 ---- #
    ax = axes[0]
    ax.plot(epochs, train_loss, color=_COLOR_TRAIN, marker="o", markersize=3, label="Train loss")
    if val_loss:
        ax.plot(epochs, val_loss, color=_COLOR_VAL, marker="s", markersize=3, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("(a) Loss curves")
    ax.legend(frameon=False)

    # ---- (b) 准确率曲线 ---- #
    ax = axes[1]
    if train_acc_pct:
        ax.plot(epochs, train_acc_pct, color=_COLOR_TRAIN, marker="o", markersize=3, label="Train acc")
    if val_acc_pct:
        ax.plot(epochs, val_acc_pct, color=_COLOR_VAL, marker="s", markersize=3, label="Val acc")
        best_epoch = int(np.argmax(val_acc_pct))
        ax.scatter([epochs[best_epoch]], [val_acc_pct[best_epoch]], color="black", zorder=5, s=30)
        ax.annotate(f"best {val_acc_pct[best_epoch]:.2f}%",
                    xy=(epochs[best_epoch], val_acc_pct[best_epoch]),
                    xytext=(5, -14), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) Accuracy curves")
    ax.legend(frameon=False)

    # ---- (c) 学习率曲线 ---- #
    ax = axes[2]
    if lr:
        ax.plot(epochs, lr, color=_COLOR_LR, marker="^", markersize=3)
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate (log scale)")
    ax.set_title("(c) LR schedule (warmup + cosine)")

    fig.suptitle(f"{title} — training curves", fontsize=13, y=1.02)
    fig.tight_layout()

    _finalize(fig, save_path, show)
    return fig


# --------------------------------------------------------------------------- #
# 2. 混淆矩阵
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
    title: str = "Confusion matrix (test set)",
    show: bool = False,
):
    """绘制混淆矩阵热力图。

    参数
    ----
    cm : np.ndarray, shape (C, C)
        ``cm[i, j]`` 表示真实类别 i 被预测为类别 j 的样本数。
    class_names : Sequence[str]
        类别名称列表（长度必须为 C）。
    save_path : str | None
        保存路径。
    normalize : bool
        True 时按真实类别（行）归一化，便于比较各类召回率。
    """
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        denom = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, np.maximum(denom, 1e-12))

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=cm.max() if cm.size else 1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Proportion" if normalize else "Count")

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    # 在每个格子中标注数值
    thresh = cm.max() * 0.6 if cm.size else 0.5
    for i in range(n):
        for j in range(n):
            value = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, value, ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    _finalize(fig, save_path, show)
    return fig


# --------------------------------------------------------------------------- #
# 3. 数据样本网格
# --------------------------------------------------------------------------- #
def plot_data_samples(
    images,
    labels,
    class_names: Sequence[str],
    save_path: Optional[str] = None,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    max_items: int = 16,
    cols: int = 8,
    show: bool = False,
):
    """把若干张训练图像拼成网格，用于报告中的"数据样例"插图。

    参数
    ----
    images : Tensor | np.ndarray
        形状 ``(N, 3, H, W)`` 的图像张量。
    labels : Tensor | np.ndarray
        对应的整数标签。
    mean, std : Sequence[float] | None
        若提供则执行反归一化，使图像颜色正确。
    """
    images = np.asarray(images)
    labels = np.asarray(labels)
    n = min(int(max_items), images.shape[0])
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx in range(rows * cols):
        ax = axes[idx]
        ax.axis("off")
        if idx >= n:
            continue
        img = images[idx]
        if img.ndim == 3 and img.shape[0] == 3:      # (C, H, W) -> (H, W, C)
            img = np.transpose(img, (1, 2, 0))
        if mean is not None and std is not None:
            img = img * np.asarray(std).reshape(1, 1, 3) + np.asarray(mean).reshape(1, 1, 3)
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(class_names[int(labels[idx])], fontsize=8)

    fig.suptitle("CIFAR-10 sample images", fontsize=12)
    fig.tight_layout()
    _finalize(fig, save_path, show)
    return fig


# --------------------------------------------------------------------------- #
# 4. 逐类准确率条形图
# --------------------------------------------------------------------------- #
def plot_per_class_accuracy(
    per_class: Sequence[float],
    class_names: Sequence[str],
    save_path: Optional[str] = None,
    title: str = "Per-class accuracy (test set)",
    show: bool = False,
):
    """绘制逐类准确率条形图（可快速定位表现最差的类别）。"""
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    values = np.asarray(per_class, dtype=np.float64) * 100.0
    bars = ax.bar(range(len(values)), values, color=_COLOR_TRAIN, alpha=0.85)

    # 在柱顶标注数值
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.5, f"{value:.1f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{title} — mean {values.mean():.2f}%")
    fig.tight_layout()
    _finalize(fig, save_path, show)
    return fig
