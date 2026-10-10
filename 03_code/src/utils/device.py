# -*- coding: utf-8 -*-
"""
设备管理与**显存自适应 batch size**。

本模块解决 Colab 上最常见的一个问题：不同 GPU（T4 15GB / L4 24GB / A100 40GB，
或免费版随时可能被降级）能承受的 batch size 不同，用户写死的 batch size 很容易 OOM。

做法：在正式训练前做一次**真实的前向 + 反向 + 优化器更新**探针（dry run），
从期望的 batch size 开始，若显存不足则逐次减半，直到能够跑通；
在 ``allow_growth=True`` 时还会向上倍增，尽量吃满显存以提高吞吐。
"""

from __future__ import annotations

import gc
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from .amp import autocast_context, device_type_of, make_grad_scaler
from .logger import get_logger

#: 探针允许占用的“可用显存”比例上限：只有 `探针峰值 <= 可用显存 × 该比例` 才允许继续放大 batch size。
#:
#: 为什么需要留余量
#: --------------
#: Windows 的 WDDM 允许显存**过度提交**：CUDA 可以分配超过物理显存的显存（超出部分落到共享内存），
#: 于是探针可能在 8 GB 卡上“成功”跑出 10 GB 的峰值，而真实训练随即 OOM 或抛出
#: ``CUDA driver error: device not ready / Failed to create GPU mapping``。
#: 留出 30% 余量（并同时考虑桌面/其它进程已占用的显存）可以把这类误判挡在训练之前。
_PROBE_FREE_RATIO: float = 0.7


# --------------------------------------------------------------------------- #
# 设备选择
# --------------------------------------------------------------------------- #
def get_device(prefer: str = "auto") -> torch.device:
    """自动选择可用设备：CUDA > Apple MPS > CPU。

    参数
    ----
    prefer : str
        "auto" 自动选择；也可显式指定 "cuda" / "mps" / "cpu"。
    """
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """生成人类可读的设备描述，用于日志与报告。"""
    dtype = device_type_of(device)
    if dtype == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        return f"CUDA:{idx} | {name} | 显存 {total:.1f} GB"
    if dtype == "mps":
        return "Apple MPS (Metal Performance Shaders)"
    return "CPU"


def empty_cache(device: torch.device) -> None:
    """释放 Python 侧未引用对象并清空 CUDA 显存缓存。"""
    gc.collect()
    if device_type_of(device) == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def gpu_total_memory_gb(device: torch.device) -> float:
    """返回 GPU 总显存（GB）；非 CUDA 设备返回 0。"""
    if device_type_of(device) != "cuda":
        return 0.0
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)


# --------------------------------------------------------------------------- #
# 显存自适应 batch size
# --------------------------------------------------------------------------- #
def _is_oom_error(err: BaseException) -> bool:
    """判断异常是否为“显存不足/显存不可用”。

    除了标准的 ``torch.cuda.OutOfMemoryError``，这里还覆盖 Windows(WDDM) 上
    显存过度提交失败时的驱动报错（``CUDA driver error: device not ready``、
    ``Failed to create GPU mapping``）。二者在语义上都是“当前 batch size 放不下”，
    因此都应该触发**下调 batch size** 而不是让训练直接崩掉。
    """
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None) if hasattr(torch, "cuda") else None
    if oom_cls is not None and isinstance(err, oom_cls):
        return True
    message = str(err).lower()
    return any(key in message for key in (
        "out of memory",
        "cuda driver error",
        "device not ready",
        "failed to create gpu mapping",
    ))


def _free_memory_mb(device) -> float:
    """返回当前可用显存（MB）；查询失败时返回 -1。"""
    try:
        free, _total = torch.cuda.mem_get_info(device)
        return float(free) / 1024 ** 2
    except Exception:  # pragma: no cover - 极少数驱动不支持
        return -1.0


def _dry_run(
    build_model_fn: Callable[[], nn.Module],
    batch_size: int,
    img_size: int,
    num_classes: int,
    device: torch.device,
    amp: bool,
) -> None:
    """用真实数据形状执行一次 前向+反向+优化器更新，以探测显存是否足够。

    探针使用与正式训练一致的模型结构、输入尺寸与精度（AMP），
    因此其显存占用可以很好地代表真实训练时的峰值占用。

    若显存不足，抛出的 OOM 异常会向上传播，由调用方捕获并下调 batch size。
    """
    model = build_model_fn().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = make_grad_scaler(device, enabled=amp)

    images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    targets = torch.randint(0, num_classes, (batch_size,), device=device)

    with autocast_context(device, enabled=amp):
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, targets)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # 主动释放探针占用的资源，避免其影响后续真实训练的显存统计
    del model, optimizer, scaler, images, targets, logits, loss
    empty_cache(device)


def resolve_batch_size(
    build_model_fn: Callable[[], nn.Module],
    img_size: int,
    num_classes: int,
    device: torch.device,
    requested_batch_size: int = 128,
    min_batch_size: int = 8,
    max_batch_size: int = 512,
    amp: bool = True,
    auto: bool = True,
    logger: Optional[object] = None,
) -> int:
    """探测并返回当前设备实际可用的 batch size。

    策略
    ----
    1. 非 CUDA 设备（CPU / MPS）不存在显存不足问题，直接返回期望值；
    2. CUDA 设备上先做一次 dry run；若显存不足（含 Windows/WDDM 的
       "CUDA driver error: device not ready" 等显存不可用错误），
       则将 batch size 减半后重试，直到成功或降到 ``min_batch_size`` 为止；
    3. 若 ``auto=True``，成功后再尝试逐级放大（最细粒度到 ``max_batch_size``），
       但要求 ``探针峰值 <= 当前可用显存 × :data:`_PROBE_FREE_RATIO```，
       即**必须为真实训练留出余量** —— 否则 WDDM 的显存过度提交会让探针
       "通过"一个物理上放不下的 batch size，真实训练随即失败。

    返回
    ----
    int
        最终可用的 batch size（保证不小于 ``min_batch_size``，除非连它都跑不通，
        此时抛出 ``RuntimeError`` 提示用户减小模型或输入尺寸）。
    """
    log = logger if logger is not None else get_logger()
    log_fn = getattr(log, "info", print)

    if device_type_of(device) != "cuda":
        log_fn(f"[batch-size] 设备为 {device_type_of(device)}，无显存限制，使用 batch size = {requested_batch_size}")
        return max(1, int(requested_batch_size))

    def can_run(bs: int) -> bool:
        """尝试以 bs 跑一次探针；成功返回 True，显存放不下返回 False。"""
        try:
            _dry_run(build_model_fn, bs, img_size, num_classes, device, amp)
            return True
        except BaseException as err:  # noqa: BLE001 - 需要区分显存问题与其他异常
            if _is_oom_error(err):
                empty_cache(device)
                return False
            raise

    def probe(bs: int) -> Tuple[bool, float, float]:
        """跑一次探针，返回 ``(是否跑通, 峰值显存 MB, 探针结束后可用显存 MB)``。

        每次都先 ``empty_cache`` 并重置峰值统计，避免上一次探针的缓存/峰值污染判定
        （否则会出现"探针峰值 10856 MB > 物理显存 8 GB"这种误导性日志）。
        """
        empty_cache(device)
        torch.cuda.reset_peak_memory_stats(device)
        ok = can_run(bs)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        empty_cache(device)
        return ok, peak_mb, _free_memory_mb(device)

    # ---- 1) 向下探测：找到能跑通的上界 ---- #
    bs = max(int(min_batch_size), min(int(requested_batch_size), int(max_batch_size)))
    log_fn(f"[batch-size] 开始显存探测：期望 {requested_batch_size}，候选 {bs}，"
           f"范围 [{min_batch_size}, {max_batch_size}]，AMP={amp}")
    found = False
    peak_mb = free_mb = -1.0
    while bs >= int(min_batch_size):
        ok, peak_mb, free_mb = probe(bs)
        if ok:
            found = True
            break
        next_bs = bs // 2
        log_fn(f"[batch-size] batch size = {bs} 显存不足或显存不可用，自动下调至 {next_bs}")
        bs = next_bs

    if not found:
        raise RuntimeError(
            f"即使 batch size = {min_batch_size} 仍显存不足，请换用更小的模型预设"
            f"（--model atto/femto/pico）或关闭 AMP 之外的其他开销。"
        )

    # ---- 2) 向上探测：在“留足余量”的前提下尽量放大 ---- #
    if auto:
        while bs * 2 <= int(max_batch_size):
            ok, next_peak_mb, next_free_mb = probe(bs * 2)
            if not ok:
                break
            if next_free_mb > 0 and next_peak_mb > next_free_mb * _PROBE_FREE_RATIO:
                log_fn(
                    f"[batch-size] batch size = {bs * 2} 探针峰值 {next_peak_mb:.0f} MB 超过可用显存 "
                    f"{next_free_mb:.0f} MB 的 {_PROBE_FREE_RATIO:.0%}（WDDM 允许过度提交，"
                    f"需为训练留余量），停止放大，最终采用 {bs}"
                )
                break
            bs *= 2
            peak_mb, free_mb = next_peak_mb, next_free_mb
            log_fn(f"[batch-size] 显存仍有富余，自动上调至 {bs}")

    log_fn(f"[batch-size] 最终确定 batch size = {bs}（探针峰值显存 {peak_mb:.0f} MB，"
           f"当前可用显存 {free_mb:.0f} MB，GPU 总显存 {gpu_total_memory_gb(device):.1f} GB）")
    return bs


def compute_accum_steps(target_batch_size: int, actual_batch_size: int) -> int:
    """计算梯度累积步数，使"等效 batch size"尽量接近用户期望值。

    例如期望 128、实际只能用 32，则累积 4 步，优化动态上仍接近 batch=128。
    """
    if actual_batch_size <= 0:
        return 1
    steps = int(round(target_batch_size / float(actual_batch_size)))
    return max(1, steps)
