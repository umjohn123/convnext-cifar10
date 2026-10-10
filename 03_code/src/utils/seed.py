# -*- coding: utf-8 -*-
"""
随机种子控制（可复现性保证）。

为保证多次运行结果一致，需要同时固定：
1. Python 内置 ``random`` 模块的种子；
2. NumPy 的随机种子（数据增强、Mixup 的 Beta 采样依赖它）；
3. PyTorch CPU / GPU 的随机种子；
4. DataLoader 多进程 worker 的随机种子（见 :func:`seed_worker`）。
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """固定所有随机源。

    参数
    ----
    seed : int
        全局随机种子。
    deterministic : bool
        为 True 时强制 cuDNN 使用确定性算法（速度略降，但结果完全可复现）；
        为 False 时允许 cuDNN 自动搜索最优卷积算法（速度更快）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # 多卡场景
    os.environ["PYTHONHASHSEED"] = str(seed)  # 影响 str/bytes 的 hash，保证 dict 顺序稳定

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch >= 1.8 提供该开关，可进一步降低非确定性算子的影响
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - 老版本兜底
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 (worker_id 由 PyTorch 传入)
    """DataLoader 多进程 worker 的初始化函数：让每个 worker 拥有独立的可复现种子。"""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """创建固定种子的 ``torch.Generator``，用于 DataLoader 的 shuffle 复现。"""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
