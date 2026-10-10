# -*- coding: utf-8 -*-
"""杂项工具：指标累加器、JSON 读写、参数量统计、时间格式化。

说明
----
CSV 类导出（``metrics.csv`` / ``history.csv``）已统一收敛到
:mod:`src.utils.csv_logger`，本模块只保留与格式无关的通用小工具。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple


class AverageMeter:
    """滑动平均指标累加器（用于按样本数加权的 epoch 平均损失/准确率）。"""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0     # 当前累计值（总和）
        self.avg = 0.0     # 平均值
        self.sum = 0.0     # 总和
        self.count = 0     # 样本计数

    def update(self, val: float, n: int = 1) -> None:
        """累加一次观测值。

        参数
        ----
        val : float
            本次观测的指标值（如某个 batch 的损失或准确率）。
        n : int
            本次观测对应的样本数量，用于加权平均。
        """
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

    def __str__(self) -> str:  # 便于格式化输出
        return f"{self.name}={self.avg:.4f}" if self.name else f"{self.avg:.4f}"


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    """把任意可序列化对象以 UTF-8 JSON 写入磁盘（自动创建父目录）。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Any:
    """读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model) -> Tuple[int, int]:
    """统计模型参数量。

    返回
    ----
    (total, trainable) : tuple[int, int]
        总参数量与可训练参数量。
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_params(num_params: int) -> str:
    """把参数量格式化为易读字符串（如 "27.83 M"）。"""
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f} M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.2f} K"
    return str(num_params)


def human_time(seconds: float) -> str:
    """把秒数格式化为 "1h 02m 03s" 形式的字符串。"""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def as_plain_dict(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """把配置字典中不可序列化的类型（如 tuple）转换为 list。"""
    out: Dict[str, Any] = {}
    for k, v in cfg_dict.items():
        if isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = as_plain_dict(v)
        else:
            out[k] = v
    return out


def optional(value: Optional[Any], default: Any) -> Any:
    """若 value 为 None 则返回 default，否则返回 value。"""
    return default if value is None else value
