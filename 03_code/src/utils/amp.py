# -*- coding: utf-8 -*-
"""
混合精度（AMP）兼容层。

PyTorch 2.4 起推荐使用 ``torch.amp.autocast(device_type=...)`` 与
``torch.amp.GradScaler(device=...)``；旧版本则使用 ``torch.cuda.amp``。
本模块屏蔽这一差异，使工程在 Colab 各版本 PyTorch 上都能直接运行。

精度选择（本项目踩过的坑）
--------------------------
低精度默认**优先 bfloat16**（Ampere 及以后原生支持），T4 等旧卡回退 fp16。
原因是 fp16 的动态范围过窄：本模型 LayerScale 初值为 ``1e-6``，18 个残差块叠加后
浅层 Block 的真实梯度只有 ``1e-6`` 甚至更低；一旦 GradScaler 的动态 scale 因
fp16 溢出被压到极小值（实测降到 ``0.0156``），梯度乘上 scale 后仍小于 fp16 的
最小次正规数（``6e-8``），反向传播会把它**四舍五入成精确的 0** ——
这些参数从此再也收不到梯度（检查点里 ``exp_avg_sq`` 恒为 0），
表现为"loss 不下降、精度停在 25%"的假收敛。
bfloat16 的阶码与 fp32 相同（8 位），不存在这一问题。
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch

# 是否可用新版 AMP API
try:
    from torch.amp import GradScaler as _NewGradScaler  # type: ignore
    from torch.amp import autocast as _new_autocast    # type: ignore

    _HAS_NEW_AMP = True
except Exception:  # pragma: no cover - 旧版本 PyTorch 兜底
    _NewGradScaler = None  # type: ignore
    _HAS_NEW_AMP = False


def device_type_of(device) -> str:
    """返回设备的类型字符串（如 "cuda" / "cpu" / "mps"）。"""
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":")[0]


def bf16_natively_supported(device) -> bool:
    """判断设备是否**原生**支持 bfloat16（计算能力 >= 8.0，即 Ampere 及以后）。

    为什么要看计算能力而不是 ``torch.cuda.is_bf16_supported()``：
    新版 PyTorch 的后者在部分设备上会因为"可以用 fp32 模拟"而返回 True，
    而模拟 bf16 在 T4（7.5）上会显著拖慢训练。这里只认原生支持。
    """
    if device_type_of(device) != "cuda":
        return False
    try:
        index = device.index if isinstance(device, torch.device) and device.index is not None \
            else torch.cuda.current_device()
        return int(torch.cuda.get_device_capability(index)[0]) >= 8
    except Exception:  # pragma: no cover - 查询失败时保守回退 fp16
        return False


def resolve_amp_dtype(device, prefer: str = "auto") -> torch.dtype:
    """选择 autocast 使用的低精度类型。

    为什么默认优先 bfloat16
    ----------------------
    fp16 的最小次正规数约为 ``6e-8``：当真实梯度小于该量级（本工程在
    LayerScale 初值 ``1e-6`` + 18 个残差块的组合下，浅层 Block 的梯度量级仅 ``1e-6``
    甚至更低）时，**即使乘上 GradScaler 的动态 scale 也会下溢成精确的 0**，
    对应参数从此再也收不到梯度（检查点中表现为 ``exp_avg_sq`` 恒为 0，
    训练日志表现为 loss 不下降、精度停在随机水平）。

    bfloat16 的指数范围与 fp32 完全相同（阶码 8 位），因此不存在这一下溢问题。
    Ampere 及以后的原生支持 bf16，故默认优先 bf16；T4 等旧卡回退 fp16，
    并由 :func:`describe_amp` / 训练日志中的 ``amp_scale`` 做显式监控。
    """
    key = str(prefer).lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16", "half"):
        return torch.float16
    return torch.bfloat16 if bf16_natively_supported(device) else torch.float16


def autocast_context(device, enabled: bool = True, dtype: Optional[torch.dtype] = None):
    """返回 autocast 上下文管理器。

    仅在 CUDA 且 ``enabled=True`` 时真正启用低精度混合精度；
    其他情况（CPU / MPS / 显式关闭）返回空上下文，逻辑上完全等价于直接前向。

    参数
    ----
    dtype : torch.dtype | None
        autocast 使用的低精度类型；None 表示按 :func:`resolve_amp_dtype` 自动选择
        （默认优先 bf16，回退 fp16）。显式传入时按其执行，便于做精度对照实验。
    """
    dtype_str = device_type_of(device)
    if not enabled or dtype_str != "cuda":
        return contextlib.nullcontext()
    use_dtype = dtype if dtype is not None else resolve_amp_dtype(device)
    if _HAS_NEW_AMP:
        return _new_autocast(device_type=dtype_str, enabled=True, dtype=use_dtype)
    return torch.cuda.amp.autocast(enabled=True, dtype=use_dtype)  # type: ignore[attr-defined]


def describe_amp(device, enabled: bool, dtype: Optional[torch.dtype] = None) -> str:
    """生成人类可读的混合精度说明（用于日志与结果文件）。"""
    if not enabled or device_type_of(device) != "cuda":
        return "False（fp32 全精度）"
    use_dtype = dtype if dtype is not None else resolve_amp_dtype(device)
    name = "bfloat16" if use_dtype == torch.bfloat16 else "float16"
    return f"True（autocast {name} + GradScaler）"


def make_grad_scaler(device, enabled: bool = True) -> Optional[object]:
    """创建 ``GradScaler``；仅在 CUDA + 启用 AMP 时返回实例，否则返回 None。"""
    if not enabled or device_type_of(device) != "cuda":
        return None
    if _HAS_NEW_AMP:
        try:
            return _NewGradScaler(device="cuda", enabled=True)  # PyTorch >= 2.4
        except TypeError:  # pragma: no cover - 参数名差异兜底
            return _NewGradScaler(enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)  # type: ignore[attr-defined]
