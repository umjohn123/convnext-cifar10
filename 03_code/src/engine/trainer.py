# -*- coding: utf-8 -*-
"""
训练引擎。

本模块实现 ConvNeXt 在 CIFAR-10 上的完整训练循环，包含：

* **优化器**：AdamW，并对权重为 1 维的参数（LayerNorm/BatchNorm 的 weight/bias、
  卷积与线性层的 bias）不施加权重衰减 —— 这是 ConvNeXt 官方训练脚本的做法；
* **学习率调度**：线性热身（warmup）+ 余弦退火（cosine decay）；
* **混合精度训练（AMP）**：CUDA 上使用 fp16 + GradScaler；
* **梯度累积**：显存不足导致 batch size 被自动调小后，
  用累积步数维持与期望 batch size 接近的优化动态；
* **梯度裁剪**：抑制偶发的梯度爆炸；
* **Mixup / CutMix**：论文训练配方中的强增强；
* **检查点管理**：保存最优与最后一次的模型权重，支持断点续训；
* **训练历史记录**：逐轮记录损失 / 准确率 / 学习率，供可视化与报告使用。
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..utils.amp import autocast_context, describe_amp, make_grad_scaler, resolve_amp_dtype
from ..utils.csv_logger import write_history_csv
from ..utils.misc import AverageMeter, human_time, save_json
from ..utils.mixup import apply_mixup_cutmix, mixup_criterion
from ..utils.visualize import plot_training_curves
from .evaluator import evaluate

#: GradScaler 动态 scale 的安全下限。低于该值说明 fp16 已频繁溢出、
#: scale 被压到极小，梯度继续乘 scale 后将大面积下溢为 0（详见 src/utils/amp.py）。
AMP_SCALE_FLOOR: float = 1.0


# --------------------------------------------------------------------------- #
# 优化器与学习率调度
# --------------------------------------------------------------------------- #
def build_optimizer(model: nn.Module, lr: float, weight_decay: float, betas=(0.9, 0.999)) -> torch.optim.AdamW:
    """构建 AdamW 优化器，并对参数做**权重衰减分组**。

    为 1 维参数（如 LayerNorm 的 weight/bias、所有 bias）单独分组，
    其 ``weight_decay=0``；其余（卷积核、线性层权重）使用设定的权重衰减。
    这一细节对 ConvNeXt 的最终精度有可观影响。
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1 维张量必然是 norm 层的缩放/偏移或 bias，不应施加权重衰减
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=float(lr), betas=betas)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: Optional[float] = None,
) -> LambdaLR:
    """构建「线性热身 + 余弦退火」学习率调度器。

    前 ``warmup_epochs`` 轮学习率从接近 0 线性升到基础学习率；
    之后按余弦曲线从基础学习率衰减到 ``min_lr``。

    参数
    ----
    base_lr : float | None
        基础学习率；为 None 时取优化器参数组中的初始 lr。
    """
    if base_lr is None:
        base_lr = optimizer.param_groups[0]["lr"]
    min_lr_ratio = float(min_lr) / float(base_lr) if base_lr > 0 else 0.0
    warmup_epochs = max(0, int(warmup_epochs))
    total_epochs = max(1, int(epochs))

    def lr_lambda(epoch: int) -> float:
        """epoch 从 0 开始计数，返回当前轮的学习率缩放系数。"""
        if warmup_epochs > 0 and epoch < warmup_epochs:
            # 线性热身：+1 保证第 0 轮的学习率不为 0
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# --------------------------------------------------------------------------- #
# 训练器
# --------------------------------------------------------------------------- #
class Trainer:
    """封装训练循环、验证、检查点保存与训练历史记录。

    参数
    ----
    cfg : Config
        全局配置对象。
    model : nn.Module
        ConvNeXt 模型。
    train_loader, val_loader : DataLoader
        训练集与验证集加载器。
    device : torch.device
        计算设备。
    logger : Logger
        日志器。
    accum_steps : int
        梯度累积步数（由显存自适应结果推导）。
    batch_size : int
        实际使用的训练 batch size（用于日志）。
    """

    def __init__(
        self,
        cfg,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        logger,
        accum_steps: int = 1,
        batch_size: int = 0,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accum_steps = max(1, int(accum_steps))
        self.batch_size = int(batch_size)

        # 仅在 CUDA 上启用 AMP
        self.amp = bool(cfg.train.amp) and device.type == "cuda"
        # autocast 精度：Ampere 及以后用 bfloat16（无梯度下溢风险），T4 等回退 fp16
        self.amp_dtype = resolve_amp_dtype(device) if self.amp else None
        # AMP 稳定性监控：GradScaler 动态 scale 的最小值 / 是否已自动回退到 fp32
        self.amp_min_scale: Optional[float] = None
        self.amp_fell_back = False
        # 标签平滑：与 Mixup/CutMix 的软标签叠加使用
        self.criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.train.label_smoothing))
        self.optimizer = build_optimizer(self.model, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.scheduler = build_scheduler(
            self.optimizer,
            epochs=cfg.train.epochs,
            warmup_epochs=cfg.train.warmup_epochs,
            min_lr=cfg.train.min_lr,
            base_lr=cfg.train.lr,
        )
        self.scaler = make_grad_scaler(device, enabled=self.amp)

        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improve = 0
        self.class_names: Optional[List[str]] = None   # 由 train.py 注入，随检查点一起保存
        self.history: Dict[str, List[float]] = {
            "epoch": [], "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "lr": [], "epoch_time": [],
            # AMP 稳定性证据：每轮 GradScaler 的动态 scale（关闭 AMP 时为 1.0）
            "amp_scale": [],
        }

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #
    @property
    def best_ckpt_path(self) -> str:
        return os.path.join(self.cfg.ckpt_dir, "best.pt")

    @property
    def last_ckpt_path(self) -> str:
        return os.path.join(self.cfg.ckpt_dir, "last.pt")

    @property
    def history_path(self) -> str:
        return os.path.join(self.cfg.log_dir, "history.json")

    @property
    def history_csv_path(self) -> str:
        return os.path.join(self.cfg.log_dir, "history.csv")

    @property
    def curves_path(self) -> str:
        return os.path.join(self.cfg.fig_dir, "training_curves.png")

    def save_history(self) -> None:
        """把训练历史落盘（``history.json`` + ``history.csv``）。

        该函数在**每一轮结束时**调用。之所以不等训练全部结束再写，
        是因为 Colab 免费实例可能中途断连：逐轮增量落盘可保证已完成的指标不丢失，
        同时 ``history.csv`` 可直接作为报告中的"逐轮指标表"。
        """
        save_json(self.history, self.history_path)
        write_history_csv(self.history, self.history_csv_path)

    def _build_checkpoint(self, epoch: int) -> Dict[str, object]:
        """打包一个完整的训练状态（模型 + 优化器 + 调度器 + 历史）。"""
        return {
            "epoch": int(epoch),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
            "best_val_acc": float(self.best_val_acc),
            "best_epoch": int(self.best_epoch),
            "history": self.history,
            "config": self.cfg.to_dict(),
            "class_names": getattr(self, "class_names", None),
            # AMP 稳定性证据（可追溯：fp16 下溢事故的判定依据）
            "amp": {
                "enabled": bool(self.amp),
                "dtype": (str(self.amp_dtype).replace("torch.", "") if self.amp_dtype else None),
                "min_scale": self.amp_min_scale,
                "fell_back_to_fp32": bool(self.amp_fell_back),
            },
        }

    def _track_amp_scale(self) -> float:
        """记录并监控 GradScaler 的动态 scale，必要时**自动回退到 fp32**。

        为什么必须监控
        --------------
        fp16 依赖 GradScaler 放大损失以避开下溢。若模型在 fp16 下频繁溢出（inf/NaN），
        scaler 会把 scale 连续减半；scale 被压到 ``0.0156`` 量级后，
        真实量级只有 ``1e-6`` 的梯度乘上 scale 后仍小于 fp16 最小次正规数 ``6e-8``，
        反向传播会将其取整为**精确的 0** —— 对应参数从此不再更新
        （检查点中 ``exp_avg_sq`` 恒为 0），训练表现为"loss 不下降、精度停在随机水平"。

        本方法每轮读取一次 scale 并记入训练历史；一旦跌破
        :data:`AMP_SCALE_FLOOR`，且配置允许（``train.amp_fallback``），
        就**自动关闭 AMP、改为 fp32 继续训练**（模型参数本就是 fp32 主副本，
        优化器状态可无缝续用），从而在不中断 Colab 会话的前提下挽救整轮训练。
        """
        if self.scaler is None:
            return 1.0
        scale = float(self.scaler.get_scale())
        if self.amp_min_scale is None or scale < self.amp_min_scale:
            self.amp_min_scale = scale
        if self.amp and bool(getattr(self.cfg.train, "amp_fallback", True)) and scale < AMP_SCALE_FLOOR:
            epoch_no = len(self.history["epoch"]) + 1
            self.logger.warning(
                f"[amp] 第 {epoch_no} 轮 GradScaler scale={scale:.3e} 已跌破安全阈值 "
                f"{AMP_SCALE_FLOOR:g}：fp16 梯度正在下溢，部分参数已停止更新"
                f"（典型表现：loss 不再下降）。现将 AMP 自动降级为 fp32 并继续训练，"
                f"后续轮次不再受 fp16 动态范围限制。"
            )
            self.amp = False
            self.amp_dtype = None
            self.scaler = None
            self.amp_fell_back = True
        return scale

    def save_checkpoint(self, path: str, epoch: int) -> None:
        """保存检查点（含模型、优化器、调度器、随机状态与训练历史）。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self._build_checkpoint(epoch), path)

    def load_checkpoint(self, path: str) -> int:
        """加载检查点并恢复训练状态，返回下一个待训练的 epoch 编号。

        兼容两种加载场景：
        * 本地存在 CUDA：普通加载；
        * 从 GPU 保存、在 CPU 上加载：使用 ``map_location`` 兜底。
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到检查点文件：{path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            except Exception as err:  # pragma: no cover - 参数组不匹配时跳过
                self.logger.warning(f"优化器状态恢复失败（将使用全新状态）：{err}")
        if "scheduler_state" in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            except Exception as err:  # pragma: no cover
                self.logger.warning(f"调度器状态恢复失败：{err}")
        if self.scaler is not None and checkpoint.get("scaler_state"):
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state"])
            except Exception:  # pragma: no cover
                pass

        self.best_val_acc = float(checkpoint.get("best_val_acc", 0.0))
        self.best_epoch = int(checkpoint.get("best_epoch", 0))
        if checkpoint.get("history"):
            self.history = checkpoint["history"]

        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.logger.info(
            f"[resume] 已从 {path} 恢复：下一轮 epoch={start_epoch}，"
            f"历史最优验证准确率={self.best_val_acc:.2f}%"
        )
        return start_epoch

    def plot_curves(self, show: bool = False) -> None:
        """按当前训练历史绘制并保存训练曲线。"""
        plot_training_curves(
            self.history,
            save_path=self.curves_path,
            title=f"ConvNeXt-{self.cfg.model.name} on CIFAR-10",
            show=show,
        )

    # ------------------------------------------------------------------ #
    # 单轮训练
    # ------------------------------------------------------------------ #
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch，返回 ``{"loss": ..., "acc": ...}``。

        关键实现点
        ----------
        * 用 ``apply_mixup_cutmix`` 判断本 batch 是否应用 Mixup/CutMix；
          未应用时返回 ``targets_a == targets_b == targets, lam=1.0``，
          因此损失函数可以统一写成 :func:`mixup_criterion` 而无需分支判断。
        * 训练准确率同样按 Mixup 的软标签计算：``lam*acc(y_a) + (1-lam)*acc(y_b)``。
        * 损失在反向传播前除以 ``accum_steps``，使累积后的梯度等价于大 batch。
        """
        self.model.train()
        loss_meter = AverageMeter("train_loss")
        acc_meter = AverageMeter("train_acc")

        # ---- 每个 epoch 的 step 上限（仅用于限时冒烟测试）---- #
        # 设置后，一个 epoch 的实际耗时只取决于 step 数，与数据集规模、硬件性能解耦；
        # 这对"只有 2 vCPU 的 Colab CPU 运行时"尤为重要。
        max_steps = self.cfg.train.max_steps_per_epoch
        total_steps = len(self.train_loader)
        num_steps = total_steps if max_steps is None else min(total_steps, max(1, int(max_steps)))

        epoch_start = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        for step, (images, targets) in enumerate(self.train_loader):
            if step >= num_steps:          # 达到上限，本 epoch 提前结束
                break
            images = images.to(self.device, non_blocking=True)
            # 强制转换为 int64，兼容不同数据集返回不同整型的情况（CrossEntropyLoss 要求 Long）
            targets = targets.to(self.device, non_blocking=True).long()

            # ---- Mixup / CutMix 数据增强 ---- #
            images, targets_a, targets_b, lam, applied = apply_mixup_cutmix(
                images, targets,
                use_mixup=bool(self.cfg.train.use_mixup),
                prob=float(self.cfg.train.mixup_prob),
                mixup_alpha=float(self.cfg.train.mixup_alpha),
                cutmix_alpha=float(self.cfg.train.cutmix_alpha),
            )

            # ---- 前向 + 损失 ---- #
            with autocast_context(self.device, enabled=self.amp, dtype=self.amp_dtype):
                outputs = self.model(images)
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                loss_for_backward = loss / self.accum_steps

            # ---- 反向 ---- #
            if self.scaler is not None:
                self.scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            # ---- 梯度累积到设定步数后统一更新参数 ---- #
            is_last_step = (step + 1) == num_steps
            if (step + 1) % self.accum_steps == 0 or is_last_step:
                if self.scaler is not None:
                    # 先 unscale 再裁剪，保证裁剪作用于真实梯度尺度
                    self.scaler.unscale_(self.optimizer)
                if float(self.cfg.train.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=float(self.cfg.train.grad_clip)
                    )
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            # ---- 统计（用原始 batch 尺寸加权） ---- #
            batch_size = targets.size(0)
            with torch.no_grad():
                preds = outputs.detach().argmax(dim=1)
                acc = (lam * (preds == targets_a).float()
                       + (1.0 - lam) * (preds == targets_b).float()).mean().item()
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)

            if (step + 1) % 50 == 0 or is_last_step:
                self.logger.info(
                    f"  epoch {epoch + 1:>3d} | step {step + 1:>4d}/{num_steps} | "
                    f"loss {loss_meter.avg:.4f} | acc {acc_meter.avg * 100:5.2f}% | "
                    f"lr {self.optimizer.param_groups[0]['lr']:.2e}"
                    + (f" | 已用时 {human_time(time.time() - epoch_start)}" if is_last_step else "")
                )

        if num_steps < total_steps:
            self.logger.info(
                f"  epoch {epoch + 1} 提前结束：本 epoch 仅训练 {num_steps}/{total_steps} 个 step"
                f"（--max-steps-per-epoch={max_steps}），因此本轮指标只代表该子集"
            )

        return {"loss": loss_meter.avg, "acc": acc_meter.avg, "time": time.time() - epoch_start}

    # ------------------------------------------------------------------ #
    # 完整训练流程
    # ------------------------------------------------------------------ #
    def fit(self) -> Dict[str, List[float]]:
        """执行完整训练流程，返回训练历史。

        每一轮的流程：训练 -> 验证 -> 更新学习率 -> 记录历史 -> 保存检查点 -> 绘制曲线。
        """
        cfg = self.cfg
        epochs = int(cfg.train.epochs)

        # ---------------- 断点续训 ---------------- #
        # --resume <路径> : 从指定检查点恢复
        # --resume auto   : 若本次实验目录下已存在 last.pt 则自动恢复（Colab 断线重连专用）
        resume_path = cfg.train.resume
        if isinstance(resume_path, str) and resume_path.strip().lower() == "auto":
            if os.path.isfile(self.last_ckpt_path):
                resume_path = self.last_ckpt_path
                self.logger.info(f"[resume] --resume auto：检测到未完成的训练，自动从 {resume_path} 续训")
            else:
                resume_path = None
                self.logger.info("[resume] --resume auto：未找到 last.pt，将从头开始训练")
        if resume_path:
            self.start_epoch = self.load_checkpoint(resume_path)
            self.save_history()   # 立刻把恢复出来的历史落盘，避免中途中断时丢失

        total_params = sum(p.numel() for p in self.model.parameters())
        if cfg.train.max_steps_per_epoch is not None:
            self.logger.warning(
                f"[train] 已启用 --max-steps-per-epoch={cfg.train.max_steps_per_epoch}："
                f"每个 epoch 只训练固定数量的 step，因此产生的指标**不代表完整 epoch**，"
                f"仅用于限时冒烟测试。"
            )
        self.logger.info("=" * 96)
        self.logger.info(f"开始训练：epochs={epochs}（本轮从 epoch {self.start_epoch + 1} 开始）| "
                         f"batch_size={self.batch_size} x accum={self.accum_steps} "
                         f"(等效 batch={self.batch_size * self.accum_steps})")
        self.logger.info(f"优化器=AdamW(lr={cfg.train.lr}, wd={cfg.train.weight_decay}) | "
                         f"调度=warmup({cfg.train.warmup_epochs}) + cosine -> {cfg.train.min_lr} | "
                         f"AMP={describe_amp(self.device, self.amp, self.amp_dtype)} | "
                         f"Mixup/CutMix={cfg.train.use_mixup} | "
                         f"参数量={total_params / 1e6:.2f}M")
        if self.amp and self.amp_dtype == torch.float16:
            self.logger.warning(
                "[amp] 当前使用 fp16（本设备不支持原生 bf16）。若每轮日志中的 "
                "amp_scale 持续低于 1，说明 fp16 梯度已在下溢、部分参数不再更新；"
                "此时建议换用支持 bf16 的 GPU，或加 --no-amp 以 fp32 训练。"
            )
        self.logger.info("=" * 96)

        for epoch in range(self.start_epoch, epochs):
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            self.logger.info(f"[Epoch {epoch + 1}/{epochs}] lr={current_lr:.3e}")

            # ---- 训练一轮 ---- #
            train_stats = self.train_one_epoch(epoch)

            # ---- 验证 ---- #
            val_metrics = evaluate(
                self.model, self.val_loader, self.criterion, self.device, amp=self.amp, topk=(1, 5)
            )

            # ---- 更新学习率（按 epoch 调度） ---- #
            self.scheduler.step()

            # ---- AMP 稳定性监控（scale 崩塌时自动降级为 fp32） ---- #
            amp_scale = self._track_amp_scale()

            # ---- 记录历史 ---- #
            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(float(train_stats["loss"]))
            self.history["train_acc"].append(float(train_stats["acc"]))
            self.history["val_loss"].append(float(val_metrics["loss"]))
            self.history["val_acc"].append(float(val_metrics["top1"]) / 100.0)
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(float(train_stats["time"]))
            self.history["amp_scale"].append(float(amp_scale))
            # 逐轮增量落盘：即使训练中途断连，已完成轮次的指标也不会丢失
            self.save_history()

            amp_note = f" | amp_scale={amp_scale:.3e}" if self.scaler is not None else ""
            self.logger.info(
                f"[Epoch {epoch + 1}/{epochs}] 完成 | "
                f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc'] * 100:.2f}% | "
                f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.2f}% "
                f"val_top5={val_metrics['top5']:.2f}% | 用时 {human_time(train_stats['time'])}{amp_note}"
            )

            # ---- 保存检查点 ---- #
            self.save_checkpoint(self.last_ckpt_path, epoch)
            if val_metrics["top1"] > self.best_val_acc:
                self.best_val_acc = float(val_metrics["top1"])
                self.best_epoch = epoch + 1
                self.epochs_without_improve = 0
                self.save_checkpoint(self.best_ckpt_path, epoch)
                self.logger.info(f"  -> 刷新最优验证准确率：{self.best_val_acc:.2f}%（已保存 best.pt）")
            else:
                self.epochs_without_improve += 1

            # ---- 周期性/最终绘制曲线 ---- #
            plot_every = int(cfg.train.plot_every)
            if plot_every > 0 and ((epoch + 1) % plot_every == 0 or (epoch + 1) == epochs):
                self.plot_curves(show=False)

            # ---- 早停 ---- #
            patience = int(cfg.train.early_stop_patience)
            if patience > 0 and self.epochs_without_improve >= patience:
                self.logger.info(
                    f"[early-stop] 连续 {patience} 轮验证准确率未提升，提前结束训练于 epoch {epoch + 1}。"
                )
                break

        # 保存训练历史（供报告与可视化复用）
        self.save_history()
        self.plot_curves(show=False)
        self.logger.info(
            f"训练结束 | 历史最优验证准确率 = {self.best_val_acc:.2f}%（epoch {self.best_epoch}）| "
            f"曲线已保存至 {self.curves_path}"
        )
        if self.amp_min_scale is not None:
            self.logger.info(
                f"[amp] GradScaler 最小 scale = {self.amp_min_scale:.3e}"
                + ("（曾因 scale 崩塌自动降级为 fp32）" if self.amp_fell_back else "")
            )
        self.logger.info(
            f"[logs] 逐轮指标已保存：{os.path.abspath(self.history_path)} 与 "
            f"{os.path.abspath(self.history_csv_path)}"
        )
        return self.history
