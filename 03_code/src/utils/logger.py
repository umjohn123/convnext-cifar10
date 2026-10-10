# -*- coding: utf-8 -*-
"""轻量日志工具：同时输出到控制台与文件，便于在 ``05_results/`` 归档训练日志。"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def get_logger(name: str = "convnext", log_file: Optional[str] = None,
               level: int = logging.INFO) -> logging.Logger:
    """获取（或创建）一个带控制台输出的 logger。

    参数
    ----
    name : str
        logger 名称，同名 logger 复用已有 handler，避免重复打印。
    log_file : str | None
        日志文件路径；为 None 时只打印到控制台。
    level : int
        日志级别。

    返回
    ----
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 避免被 root logger 重复输出

    if logger.handlers:      # 已经初始化过，直接复用
        return logger

    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
