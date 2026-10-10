# -*- coding: utf-8 -*-
"""
ConvNeXt 模型定义（适配 CIFAR-10 的 32x32 输入）。

参考文献
--------
Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie.
"A ConvNet for the 2020s". CVPR 2022. arXiv:2201.03545.
官方实现: https://github.com/facebookresearch/ConvNeXt

论文关键设计（本实现全部保留）
------------------------------
1. **宏观结构**：四个阶段，堆叠比例由 ResNet 的 (3,4,6,3) 改为 (3,3,9,3)，通道数逐阶段翻倍。
2. **卷积式 Stem**：用卷积（而非 Transformer 的 patch embedding + attention）完成 patchify。
3. **倒置瓶颈（Inverted Bottleneck）**：``1x1 升维 4 倍 -> 深度卷积 -> 1x1 降维``。
4. **大卷积核深度卷积**：深度卷积核为 **7x7**，以扩大有效感受野。
5. **ResNeXt 化**：深度卷积（groups = channels）替代普通卷积，显著降低计算量。
6. **微观设计**：GELU 替代 ReLU、减少激活层数量（每个 Block 仅 1 个 GELU）、
   LayerNorm 替代 BatchNorm、阶段之间使用**独立的下采样层**（LayerNorm + 2x2 卷积）。
7. **训练稳定性技巧**：Layer Scale（可学习缩放，初值 1e-6）与 Stochastic Depth（DropPath）。

针对 CIFAR-10 的适配
--------------------
ImageNet 的输入为 224x224，论文 Stem 为 4x4、stride=4 的 patchify 卷积，
特征图序列为 56 -> 28 -> 14 -> 7。
CIFAR-10 图像只有 32x32，若沿用 stride=4，特征图会迅速缩到 8 -> 4 -> 2 -> 1，
空间信息损失过大。因此本实现默认使用 **3x3、stride=1 的卷积式 Stem**
（``stem_mode="conv"``），四阶段分辨率变为 32 -> 16 -> 8 -> 4，
在保留论文全部关键模块的前提下适配小分辨率输入。
将 ``stem_mode`` 设为 ``"patchify"`` 即可切换回论文原版 4x4/s4 的 Stem。

速度与精度的取舍：模型容量主要由 ``depths``（每个阶段的 Block 数）与 ``dims``（通道数）
决定，可通过 :data:`CONVNEXT_PRESETS` 中的不同规模进行消融实验。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


# --------------------------------------------------------------------------- #
# 1. 基础组件
# --------------------------------------------------------------------------- #
class LayerNorm(nn.Module):
    """支持 ``channels_last`` 与 ``channels_first`` 两种数据格式的 LayerNorm。

    PyTorch 原生 ``nn.LayerNorm`` 默认只对最后一维做归一化（即 channels_last）。
    ConvNeXt 模型中张量主体是 ``(N, C, H, W)`` 的 channels_first 格式，
    因此这里提供 channels_first 的等价实现（对通道维做归一化）。
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ("channels_last", "channels_first"):
            raise NotImplementedError(f"不支持的 data_format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # channels_first：手写归一化，避免频繁 permute 带来的额外拷贝
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic Depth（逐样本随机丢弃整个残差分支）。

    论文写法（亦见 ``timm`` 实现）：训练时以 ``drop_prob`` 的概率把某个样本的
    残差分支整体置零，并把保留下来的样本除以 ``1 - drop_prob`` 以保持期望不变。
    """
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # 每个样本一个独立的伯努利掩码，其余维度广播
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x.div(keep_prob) * mask


class DropPath(nn.Module):
    """``drop_path`` 的 ``nn.Module`` 封装。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"


class ConvNeXtBlock(nn.Module):
    """ConvNeXt 的基本模块：深度卷积 + 倒置瓶颈（Inverted Bottleneck）。

    结构（对应论文 Figure 4 的最终设计）::

        x ---------------------------------- (残差)
        |
        DepthwiseConv2d(7x7, groups=C)   <- 大卷积核，扩大感受野
        LayerNorm (channels_last)
        Linear(C -> 4C)                  <- 倒置瓶颈：升维 4 倍
        GELU                             <- 全块仅有 1 个激活函数
        Linear(4C -> C)                  <- 降维回原通道数
        Layer Scale (gamma)
        |
        + ---------------------------------- (相加)

    参数
    ----
    dim : int
        输入/输出通道数。
    drop_path : float
        该 Block 的 Stochastic Depth 概率。
    layer_scale_init_value : float
        Layer Scale 初始值；<=0 表示不使用 Layer Scale。
    kernel_size : int
        深度卷积核大小，论文为 7。
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        kernel_size: int = 7,
    ):
        super().__init__()
        # 7x7 深度卷积：groups=dim，参数量仅 dim*k*k
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        # 倒置瓶颈：升维 4 倍，再降维（用 Linear 等价于 1x1 卷积）
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0 else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)                                  # (N, C, H, W)
        x = x.permute(0, 2, 3, 1)                           # -> (N, H, W, C)，便于用 LayerNorm/Linear
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)                           # -> (N, C, H, W)
        return residual + self.drop_path(x)


class ConvNeXtDownsample(nn.Module):
    """阶段之间的**独立下采样层**：LayerNorm + 2x2、stride=2 卷积。

    论文指出：使用独立的下采样层（而非像 ResNet 那样在残差分支首层做 stride=2）
    可以进一步提升精度。
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


# --------------------------------------------------------------------------- #
# 2. 模型预设
# --------------------------------------------------------------------------- #
#: 不同规模的 ConvNeXt 预设。
#: ``tiny`` 对应论文 ConvNeXt-T 的原始配置 (depths=(3,3,9,3), dims=(96,192,384,768))；
#: 其余为更轻量的规模（源自 ConvNeXt V2 的设计），便于在 Colab 免费 GPU 上快速验证。
CONVNEXT_PRESETS: Dict[str, Dict[str, Sequence[int]]] = {
    "atto":  {"depths": (2, 2, 6, 2), "dims": (40, 80, 160, 320)},
    "femto": {"depths": (2, 2, 6, 2), "dims": (48, 96, 192, 384)},
    "pico":  {"depths": (2, 2, 6, 2), "dims": (64, 128, 256, 512)},
    "nano":  {"depths": (2, 2, 8, 2), "dims": (80, 160, 320, 640)},
    "tiny":  {"depths": (3, 3, 9, 3), "dims": (96, 192, 384, 768)},
}


# --------------------------------------------------------------------------- #
# 3. 主模型
# --------------------------------------------------------------------------- #
class ConvNeXt(nn.Module):
    """面向 CIFAR-10 的 ConvNeXt 分类网络（纯卷积，不含任何注意力机制）。

    参数
    ----
    in_chans : int
        输入通道数，RGB 图像为 3。
    num_classes : int
        分类类别数，CIFAR-10 为 10。
    depths : Sequence[int]
        每个阶段的 Block 数量，论文为 (3, 3, 9, 3)。
    dims : Sequence[int]
        每个阶段的通道数，论文 Tiny 为 (96, 192, 384, 768)。
    drop_path_rate : float
        Stochastic Depth 的最大丢弃率；各 Block 的概率由 0 线性递增到该值。
    layer_scale_init_value : float
        Layer Scale 的初始值。
    stem_mode : str
        ``"conv"``   : 3x3、stride=1 卷积 Stem（适配 32x32，默认）；
        ``"patchify"``: 4x4、stride=4 卷积 Stem（论文原版，适用于大分辨率输入）。
    head_init_scale : float
        分类头初始权重的缩放系数，用于稳定训练早期的输出尺度。
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 10,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        stem_mode: str = "conv",
        head_init_scale: float = 1.0,
    ):
        super().__init__()
        assert len(depths) == 4 and len(dims) == 4, "ConvNeXt 需要 4 个阶段"
        self.depths = list(depths)
        self.dims = list(dims)
        self.stem_mode = stem_mode

        # ---------------- Stem / 下采样层 ---------------- #
        self.downsample_layers = nn.ModuleList()
        if stem_mode == "patchify":
            # 论文原版：4x4、stride=4 卷积完成 patchify
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        elif stem_mode == "conv":
            # 小分辨率适配版：3x3、stride=1 卷积，保持空间分辨率
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        else:
            raise ValueError(f"未知的 stem_mode: {stem_mode}（可选 'conv' / 'patchify'）")

        self.downsample_layers.append(stem)
        for stage_idx in range(3):
            self.downsample_layers.append(
                ConvNeXtDownsample(dims[stage_idx], dims[stage_idx + 1])
            )

        # ---------------- 四个阶段的 Block ---------------- #
        # Stochastic Depth：概率在各 Block 之间从 0 线性递增到 drop_path_rate
        total_blocks = int(sum(depths))
        dp_rates = torch.linspace(0.0, float(drop_path_rate), total_blocks).tolist()

        self.stages = nn.ModuleList()
        cursor = 0
        for stage_idx in range(4):
            blocks = [
                ConvNeXtBlock(
                    dim=dims[stage_idx],
                    drop_path=dp_rates[cursor + block_idx],
                    layer_scale_init_value=layer_scale_init_value,
                )
                for block_idx in range(depths[stage_idx])
            ]
            self.stages.append(nn.Sequential(*blocks))
            cursor += depths[stage_idx]

        # ---------------- 分类头 ---------------- #
        # 官方实现用 GroupNorm(1, C) 作为末端归一化，等价于对每个样本做 channels_first LayerNorm
        self.norm = nn.GroupNorm(1, dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        # 权重初始化
        self.apply(self._init_weights)
        # 分类头单独缩放（论文中用于稳定输出尺度）
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0.0)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """按论文/官方实现初始化权重：截断正态 N(0, 0.02)，偏置置零。"""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征图（返回下采样后的最后一层特征，形状 ``(N, C, H, W)``）。"""
        for stage_idx in range(4):
            x = self.downsample_layers[stage_idx](x)
            x = self.stages[stage_idx](x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：特征提取 -> 全局平均池化 -> 线性分类头。"""
        x = self.forward_features(x)
        x = x.mean(dim=(-2, -1))          # 全局平均池化 (N, C, H, W) -> (N, C)
        return self.head(x)               # (N, num_classes)


# --------------------------------------------------------------------------- #
# 4. 工厂函数
# --------------------------------------------------------------------------- #
def build_convnext(model_cfg, num_classes: int = 10) -> ConvNeXt:
    """根据 :class:`configs.ModelConfig` 构建 ConvNeXt 模型。

    若 ``model_cfg.depths`` / ``model_cfg.dims`` 为 None，则使用 ``model_cfg.name``
    对应的预设；否则以显式指定的结构为准（便于做结构消融实验）。

    参数
    ----
    model_cfg : ModelConfig
        模型配置对象。
    num_classes : int
        分类类别数。

    返回
    ----
    ConvNeXt
    """
    name = str(model_cfg.name).lower()
    if model_cfg.depths is not None and model_cfg.dims is not None:
        depths: Tuple[int, ...] = tuple(model_cfg.depths)
        dims: Tuple[int, ...] = tuple(model_cfg.dims)
    else:
        if name not in CONVNEXT_PRESETS:
            raise KeyError(f"未知的 ConvNeXt 预设 '{name}'，可选：{list(CONVNEXT_PRESETS)}")
        preset = CONVNEXT_PRESETS[name]
        depths = tuple(preset["depths"])
        dims = tuple(preset["dims"])

    return ConvNeXt(
        in_chans=3,
        num_classes=num_classes,
        depths=depths,
        dims=dims,
        drop_path_rate=float(model_cfg.drop_path_rate),
        layer_scale_init_value=float(model_cfg.layer_scale_init_value),
        stem_mode=str(model_cfg.stem_mode),
        head_init_scale=float(model_cfg.head_init_scale),
    )


def convnext_model_summary(model_cfg, num_classes: int = 10, input_size: int = 32) -> str:
    """生成模型结构摘要字符串（含各阶段维度与参数量），用于日志与报告。"""
    name = str(model_cfg.name).lower()
    if model_cfg.depths is not None and model_cfg.dims is not None:
        depths, dims = list(model_cfg.depths), list(model_cfg.dims)
    else:
        preset = CONVNEXT_PRESETS[name]
        depths, dims = list(preset["depths"]), list(preset["dims"])

    lines = [
        f"ConvNeXt-{name} | stem={model_cfg.stem_mode} | drop_path={model_cfg.drop_path_rate}",
        f"  input            : 3 x {input_size} x {input_size}",
    ]
    resolution = input_size if model_cfg.stem_mode == "conv" else input_size // 4
    for stage_idx in range(4):
        lines.append(
            f"  stage {stage_idx + 1}          : {resolution:>2} x {resolution:>2} x {dims[stage_idx]:<4}"
            f" | blocks={depths[stage_idx]}"
        )
        if stage_idx < 3:
            resolution = max(1, resolution // 2)
    lines.append(f"  head             : global average pool -> Linear({dims[-1]} -> {num_classes})")
    return "\n".join(lines)
