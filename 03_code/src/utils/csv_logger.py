# -*- coding: utf-8 -*-
"""
CSV 导出工具：把训练过程的逐轮指标与最终结果汇总写成 CSV。

为什么需要 CSV
--------------
报告中的表格（逐轮训练指标表、多实验对比表）若从 JSON 手工誊抄，既费时又易出错。
本模块把结构化结果**自动**导出为**列顺序稳定**的 CSV，从而：

* ``history.csv`` —— 一行一轮的训练历史，可直接作为报告"逐轮指标表"；
* ``metrics.csv`` —— 一行一次实验的结果汇总（含逐类准确率），
  把多个实验的该文件**纵向拼接**即可得到对比 / 消融表格。

由于列顺序由下面的常量显式固定（而非按字典自动展开），
不同实验、不同代码版本产出的 CSV 可以直接堆叠，不存在列错位风险。
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Sequence, Tuple


def _ensure_parent(path: str) -> None:
    """确保 CSV 的父目录存在。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. 逐轮训练历史 -> history.csv
# --------------------------------------------------------------------------- #
#: ``history.csv`` 的列定义：(表头, history 字典中的键)。
_HISTORY_COLUMNS: List[Tuple[str, str]] = [
    ("epoch",         "epoch"),
    ("train_loss",    "train_loss"),
    ("train_acc",     "train_acc"),
    ("val_loss",      "val_loss"),
    ("val_acc",       "val_acc"),
    ("lr",            "lr"),
    ("epoch_time_sec", "epoch_time"),
    # AMP 稳定性证据：fp16 的 GradScaler 动态 scale。正常应稳定在 65536 附近；
    # 若持续低于 1，说明梯度已大面积下溢、部分参数停止更新（见 src/utils/amp.py）。
    # 未启用 AMP、或已自动降级为 fp32 的轮次记为 1.0（意为"未做放大"）；
    # 崩塌发生的那一轮会如实记下极小值，并另有 amp.fell_back_to_fp32 标记。
    ("amp_scale",     "amp_scale"),
]


def _format_history_value(key: str, value: Any) -> str:
    """按列类型格式化单个取值。

    * 准确率统一输出为**百分数**（保留 4 位小数），便于直接进报告表格；
      因此既接受 0~1 的小数，也接受已经是百分数的输入。
    * 学习率用科学计数法，便于观察 warmup + cosine 的衰减轨迹。
    """
    if value is None or value == "":
        return ""
    number = float(value)
    if key in ("train_acc", "val_acc"):
        return f"{number * 100.0:.4f}" if number <= 1.0 else f"{number:.4f}"
    if key in ("lr", "amp_scale"):
        return f"{number:.3e}"
    if key == "epoch_time":
        return f"{number:.2f}"
    if key == "epoch":
        return str(int(number))
    return f"{number:.6f}"


def write_history_csv(history: Dict[str, Sequence[float]], path: str) -> str:
    """把逐轮训练历史写成 CSV（一行一轮）。

    参数
    ----
    history : dict
        训练历史字典，通常包含
        ``epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time, amp_scale``。
        缺列或各列长度不一致时会自动补空值，不会抛异常。
    path : str
        输出 CSV 路径（自动创建父目录）。

    返回
    ----
    str
        实际写入的路径。
    """
    series: Dict[str, List[Any]] = {
        key: list(history.get(key) or []) for _, key in _HISTORY_COLUMNS
    }
    num_rows = max((len(values) for values in series.values()), default=0)
    for values in series.values():
        values.extend([""] * (num_rows - len(values)))

    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([header for header, _ in _HISTORY_COLUMNS])
        for row_index in range(num_rows):
            writer.writerow([
                _format_history_value(key, series[key][row_index])
                for _, key in _HISTORY_COLUMNS
            ])
    return path


# --------------------------------------------------------------------------- #
# 2. 结果汇总 -> metrics.csv
# --------------------------------------------------------------------------- #
#: ``metrics.csv`` 的列定义：(列名, 在结果字典中的取值路径)。
_METRIC_COLUMNS: List[Tuple[str, Tuple[str, ...]]] = [
    ("experiment",            ("experiment",)),
    ("algorithm",             ("algorithm",)),
    ("model_preset",          ("architecture", "preset")),
    ("stem_mode",             ("architecture", "stem_mode")),
    ("drop_path_rate",        ("architecture", "drop_path_rate")),
    ("num_params",            ("architecture", "num_params")),
    ("dataset",               ("dataset", "name")),
    ("split",                 ("dataset", "split")),
    ("is_full_scale",         ("dataset", "is_full_scale")),
    ("train_subset_size",     ("dataset", "train_subset_size")),
    ("test_subset_size",      ("dataset", "test_subset_size")),
    ("val_ratio",             ("dataset", "val_ratio")),
    ("data_root",             ("dataset", "data_root")),
    ("device",                ("device",)),
    ("gpu_total_memory_gb",   ("gpu_total_memory_gb",)),
    ("batch_size_requested",  ("batch_size_requested",)),
    ("batch_size_actual",     ("batch_size_actual",)),
    ("grad_accum_steps",      ("grad_accum_steps",)),
    ("effective_batch_size",  ("effective_batch_size",)),
    ("epochs",                ("epochs",)),
    ("epochs_completed",      ("epochs_completed",)),
    ("best_val_acc",          ("best_val_acc",)),
    ("best_val_epoch",        ("best_val_epoch",)),
    ("test_loss",             ("test_metrics", "loss")),
    ("test_top1_acc",         ("test_metrics", "top1_acc")),
    ("test_top5_acc",         ("test_metrics", "top5_acc")),
    ("test_num_samples",      ("test_metrics", "num_samples")),
    ("macro_accuracy",        ("macro_accuracy",)),
    ("total_training_time",   ("total_training_time",)),
    ("seed",                  ("seed",)),
    # 混合精度可追溯性（fp16 下溢事故的判定依据，详见 src/utils/amp.py）：
    #   amp_dtype     —— autocast 实际使用的低精度类型（bfloat16 / float16 / 空）
    #   amp_min_scale —— GradScaler 动态 scale 的最小值；正常应接近 65536，
    #                    若为 <1 的小值则说明该次运行的骨干网络可能已停止更新
    ("amp_dtype",             ("amp", "dtype")),
    ("amp_min_scale",         ("amp", "min_scale")),
]


def _dig(data: Any, keys: Tuple[str, ...], default: Any = "") -> Any:
    """按 ``keys`` 路径逐层读取嵌套字典；任一层缺失或为 None 时返回 ``default``。"""
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current or current[key] is None:
            return default
        current = current[key]
    return current


def write_metrics_csv(summary: Dict[str, Any], path: str) -> str:
    """把一次实验的结果汇总写成单行 ``metrics.csv``（含逐类准确率列）。

    参数
    ----
    summary : dict
        与 ``metrics/test_metrics.json`` 结构一致的结果汇总字典。
        本函数对缺失字段容错（取空字符串），因此也兼容 ``evaluate.py`` 产出的
        ``eval_metrics.json``。
    path : str
        输出 CSV 路径（自动创建父目录）。

    返回
    ----
    str
        实际写入的路径。

    说明
    ----
    采用"一行一次实验"的宽表格式：前 30 列是实验设置与总体指标，
    其后追加 ``acc_<类别名>`` 逐类准确率列。因此把多个实验的 ``metrics.csv``
    纵向拼接即可直接得到报告中的对比/消融表格。
    """
    columns = [name for name, _ in _METRIC_COLUMNS]
    row = [_dig(summary, keys) for _, keys in _METRIC_COLUMNS]

    per_class = summary.get("per_class_accuracy") if isinstance(summary, dict) else None
    if isinstance(per_class, dict):
        for class_name, accuracy in per_class.items():
            columns.append(f"acc_{class_name}")
            row.append(accuracy)

    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerow(row)
    return path


def read_metrics_csv(path: str) -> List[Dict[str, str]]:
    """读取 ``metrics.csv`` 为字典列表（供后续做实验对比表使用）。"""
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
