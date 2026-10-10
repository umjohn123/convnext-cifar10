# -*- coding: utf-8 -*-
"""
合成数据集（**零下载**）：用于在无网络或网络缓慢时验证整条训练流水线。

设计目标
--------
1. **完全离线**：不访问网络、不依赖任何已下载文件，构造即用 ——
   这是"快速冒烟测试"能在数十秒内完成的关键（详见 ``--dataset synthetic``）。
2. **形态与 CIFAR-10 完全一致**：32×32×3、10 类、返回 ``(PIL.Image, int)``，
   因此可以直接复用同一套数据变换、同一个模型与同一套评估/可视化代码，
   验证的是**真实代码路径**而非另写一条"假路径"。
3. **确定性**：由固定随机种子一次性生成固定的张量集合，多次运行结果一致，可复现。
4. **可学习**：并非纯噪声 —— 每个类别叠加一个固定的"类模板"信号再混入噪声，
   使模型在小规模下也能真的学到东西。这样损失/准确率曲线会呈现**正常下降趋势**，
   从而验证"指标会动、绘图正常、检查点会刷新"；若用纯随机标签，
   曲线只能证明"代码不报错"，无法证明"训练有效"。

与真实数据集的关系
------------------
* 合成数据的指标**没有科学意义**，绝不能写入报告的实验结果；
* 它只用于环境自检与流水线冒烟测试（对应协作计划中的"先打通 MVP"原则）。
  结果文件中通过 ``dataset.name = "synthetic"`` 与 ``is_full_scale = false`` 双重标记。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .cifar10 import CIFAR10_CLASSES, IMAGE_SIZE


class SyntheticCIFAR(Dataset):
    """与 CIFAR-10 同形态的合成图像数据集（本地生成，零下载）。

    参数
    ----
    size : int
        样本数量。
    num_classes : int
        类别数量，默认 10（与 CIFAR-10 一致）。
    image_size : int
        图像边长，默认 32。
    seed : int
        生成种子。相同种子 + 相同参数 ⇒ 完全相同的图像与标签（可复现）。
    template_weight : float
        类模板在单张图像中的权重（0~1）。越大类间越易区分、任务越简单。
        默认 0.55，使模型在少量轮次内即可看到明显优于随机的准确率。
    transform : callable | None
        可选变换。本工程通常传 None，由 ``TransformSubset`` 叠加训练/评估变换。
    """

    def __init__(
        self,
        size: int = 4096,
        num_classes: int = 10,
        image_size: int = IMAGE_SIZE,
        seed: int = 1234,
        template_weight: float = 0.55,
        transform=None,
    ) -> None:
        self.size = int(size)
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.transform = transform
        self.classes: List[str] = list(CIFAR10_CLASSES[: self.num_classes])

        rng = np.random.RandomState(self.seed)

        # 1) 每个类别一个固定的"类模板"（低频结构信号）
        templates = rng.randint(0, 256, size=(self.num_classes, image_size, image_size, 3))
        templates = templates.astype(np.float32)

        # 2) 逐样本随机标签，并按标签取模板
        labels = rng.randint(0, self.num_classes, size=self.size).astype(np.int64)
        template = templates[labels]                       # (N, H, W, 3)

        # 3) 混入均匀噪声：模板权重越高越易分类
        noise = rng.randint(0, 256, size=(self.size, image_size, image_size, 3)).astype(np.float32)
        weight = float(template_weight)
        data = np.clip(weight * template + (1.0 - weight) * noise, 0.0, 255.0)

        # 与 CIFAR-10 保持一致：uint8、HWC 布局，便于复用同一套 PIL 变换
        self.data: np.ndarray = data.astype(np.uint8)
        self.targets: List[int] = labels.tolist()

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[object, int]:
        """返回 ``(PIL.Image, int)``，与 ``torchvision.datasets.CIFAR10`` 完全一致。"""
        image = Image.fromarray(self.data[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[index])

    def extra_repr(self) -> str:
        return (f"size={self.size}, num_classes={self.num_classes}, "
                f"image_size={self.image_size}, seed={self.seed}")


def build_synthetic_dataset(cfg, train: bool, logger=None) -> Dataset:
    """按配置构造合成数据集（训练/测试两份，种子不同以免完全重叠）。

    参数
    ----
    cfg : Config
        读取 ``data.synthetic_train_size`` / ``data.synthetic_test_size`` /
        ``data.num_classes`` 与 ``train.seed``。
    train : bool
        True 返回训练用合成集，False 返回测试用合成集。
    """
    size = int(cfg.data.synthetic_train_size if train else cfg.data.synthetic_test_size)
    seed = int(cfg.train.seed) + (0 if train else 10_000)
    dataset = SyntheticCIFAR(
        size=size,
        num_classes=int(cfg.data.num_classes),
        image_size=IMAGE_SIZE,
        seed=seed,
        transform=None,
    )
    if logger is not None:
        logger.warning(
            f"[data] 使用**合成数据**（dataset=synthetic，{size} 张，seed={seed}）——"
            f"零下载、仅用于流程冒烟测试；其指标没有科学意义，禁止写入报告结果。"
        )
    return dataset
