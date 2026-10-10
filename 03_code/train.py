# -*- coding: utf-8 -*-
"""
ConvNeXt + CIFAR-10 训练入口脚本。

用法（Google Colab 中直接运行）::

    !python train.py --epochs 60 --model tiny --batch-size 128

以「实验隔离」方式运行（推荐：不同实验各自独立成目录，互不覆盖）::

    !python train.py --run-tag full_tiny_60e --epochs 60 --model tiny
    !python train.py --run-tag ablation_patchify_stem --model pico --stem-mode patchify
    !python train.py --run-tag seed43 --model pico --epochs 30 --seed 43

完整流程
--------
1. 解析配置、创建输出目录、开启日志；
2. 固定随机种子、选择计算设备（CUDA / MPS / CPU）；
3. **显存自适应**探测 batch size（不足则自动下调，并用梯度累积补偿等效 batch）；
4. 构建 CIFAR-10 数据加载器（训练 45000 / 验证 5000 / 测试 10000）；
5. 构建 ConvNeXt、AdamW、warmup + cosine 调度器，执行训练；
6. 载入最优检查点在**测试集**上评估，输出准确率、逐类准确率与混淆矩阵；
7. 绘制并保存损失曲线、准确率曲线与混淆矩阵图。

产物（默认写入 ``./outputs/``，按"结果三分类"组织）
--------------------------------------------------
* ``models/best.pt``    —— 【模型】验证集最优模型（含优化器/调度器/历史/配置）
* ``models/last.pt``    —— 【模型】最后一轮的模型（可用于断点续训）
* ``logs/train.log``    —— 【日志】完整训练日志（带时间戳）
* ``logs/history.json`` —— 【日志】逐轮损失 / 准确率 / 学习率 / 耗时
* ``figures/training_curves.png``      —— 【图表】损失与准确率曲线
* ``figures/confusion_matrix.png``     —— 【图表】测试集混淆矩阵
* ``figures/per_class_accuracy.png``   —— 【图表】逐类准确率
* ``figures/data_samples.png``         —— 【图表】数据样例
* ``metrics/test_metrics.json``        —— 【指标】最终测试指标
* ``metrics/metrics.csv``              —— 【指标】一行式指标汇总表（便于拼接成对比/消融表）
* ``metrics/config.json``              —— 【指标】本次运行的全部超参数（可复现性证据）

实验目录隔离
------------
指定 ``--run-tag <name>`` 后，上述全部产物写入 ``<output-dir>/<name>/`` 而非 ``<output-dir>/``，
从而保证「全量训练 / 对比实验 / 消融实验 / 多种子重复」等多次运行**互不覆盖、互不交叉引用**。
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from configs import parse_args
from src.data import build_dataloaders
from src.engine.evaluator import analyze_predictions, collect_predictions, evaluate
from src.engine.trainer import Trainer
from src.models.convnext import build_convnext, convnext_model_summary
from src.utils.device import (
    compute_accum_steps,
    describe_device,
    get_device,
    gpu_total_memory_gb,
    resolve_batch_size,
)
from src.utils.logger import get_logger
from src.utils.csv_logger import write_metrics_csv
from src.utils.misc import count_parameters, format_params, human_time, save_json
from src.utils.seed import set_seed
from src.utils.tta import describe_tta
from src.utils.visualize import plot_confusion_matrix, plot_data_samples, plot_per_class_accuracy

_IMAGE_SIZE = 32  # CIFAR-10 图像尺寸，供显存探测与模型摘要使用


def main() -> None:
    """训练主流程。"""
    wall_start = time.time()

    # ------------------------------------------------------------------ #
    # 1. 配置与日志
    # ------------------------------------------------------------------ #
    cfg = parse_args()
    cfg.make_dirs()
    logger = get_logger(
        name="convnext_train",
        log_file=os.path.join(cfg.log_dir, "train.log"),
    )
    logger.info("=" * 96)
    logger.info(f"实验名称：{cfg.exp_name}")
    logger.info(f"输出根目录：{os.path.abspath(cfg.output_dir)}")
    if cfg.run_tag:
        logger.info(f"实验隔离标签：{cfg.run_tag} -> 本次产物写入 {os.path.abspath(cfg.run_root)}")
    else:
        logger.info(f"未设置实验标签（--run-tag），产物直接写入 {os.path.abspath(cfg.run_root)}")
    cfg.save(os.path.join(cfg.result_dir, "config.json"))
    logger.info("=" * 96)

    # ------------------------------------------------------------------ #
    # 2. 随机种子与设备
    # ------------------------------------------------------------------ #
    set_seed(seed=int(cfg.train.seed), deterministic=bool(cfg.train.deterministic))
    device = get_device(cfg.device)
    logger.info(f"[device] {describe_device(device)}")
    logger.info(f"[seed] 全局随机种子 = {cfg.train.seed} | 确定性模式 = {cfg.train.deterministic}")

    # ------------------------------------------------------------------ #
    # 3. 显存自适应 batch size
    # ------------------------------------------------------------------ #
    def build_model() -> torch.nn.Module:
        """模型构建闭包（供显存探测与正式训练复用同一结构）。"""
        return build_convnext(cfg.model, num_classes=cfg.data.num_classes)

    actual_batch_size = resolve_batch_size(
        build_model_fn=build_model,
        img_size=_IMAGE_SIZE,
        num_classes=int(cfg.data.num_classes),
        device=device,
        requested_batch_size=int(cfg.train.batch_size),
        min_batch_size=int(cfg.train.min_batch_size),
        max_batch_size=int(cfg.train.max_batch_size),
        amp=bool(cfg.train.amp),
        auto=bool(cfg.train.auto_batch_size),
        logger=logger,
    )
    # 若 batch size 被自动调小，则用梯度累积把等效 batch 补回期望值
    accum_steps = compute_accum_steps(int(cfg.train.batch_size), actual_batch_size)
    if actual_batch_size != int(cfg.train.batch_size):
        logger.info(
            f"[batch-size] 期望 {cfg.train.batch_size} -> 实际 {actual_batch_size}，"
            f"采用梯度累积 {accum_steps} 步（等效 batch = {actual_batch_size * accum_steps}）"
        )

    # ------------------------------------------------------------------ #
    # 4. 数据
    # ------------------------------------------------------------------ #
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        cfg, batch_size=actual_batch_size, logger=logger
    )

    # 保存一批数据样例图（用于报告中的"数据集说明"）
    try:
        sample_images, sample_labels = next(iter(val_loader))
        plot_data_samples(
            sample_images[:16].cpu().numpy(),
            sample_labels[:16].cpu().numpy(),
            class_names,
            save_path=os.path.join(cfg.fig_dir, "data_samples.png"),
            mean=cfg.data.mean,
            std=cfg.data.std,
        )
        logger.info(f"[figures] 数据样例图已保存：{os.path.join(cfg.fig_dir, 'data_samples.png')}")
    except Exception as err:  # pragma: no cover - 绘图失败不应中断训练
        logger.warning(f"绘制数据样例图失败（不影响训练）：{err}")

    # ------------------------------------------------------------------ #
    # 5. 模型
    # ------------------------------------------------------------------ #
    model = build_model()
    total_params, trainable_params = count_parameters(model)
    logger.info("[model] 结构摘要：")
    for line in convnext_model_summary(cfg.model, num_classes=cfg.data.num_classes,
                                       input_size=_IMAGE_SIZE).split("\n"):
        logger.info(f"    {line}")
    logger.info(
        f"[model] 总参数量 = {format_params(total_params)}（{total_params:,}）| "
        f"可训练参数量 = {format_params(trainable_params)}"
    )

    # ------------------------------------------------------------------ #
    # 6. 训练
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
        accum_steps=accum_steps,
        batch_size=actual_batch_size,
    )
    trainer.class_names = class_names
    history = trainer.fit()

    # ------------------------------------------------------------------ #
    # 7. 载入最优权重并在测试集上评估
    # ------------------------------------------------------------------ #
    logger.info("=" * 96)
    logger.info("开始在**测试集**上评估验证集最优模型（best.pt）……")
    best_ckpt = trainer.best_ckpt_path
    if os.path.isfile(best_ckpt):
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
        trainer.model.load_state_dict(checkpoint["model_state"])
        logger.info(f"[evaluate] 已载入 {best_ckpt}（对应 epoch {checkpoint.get('epoch', -1) + 1}）")
    else:
        logger.warning("[evaluate] 未找到 best.pt，使用当前（最后一轮）权重进行评估")

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(cfg.train.label_smoothing))
    use_tta = bool(cfg.train.tta)
    if use_tta:
        logger.info(f"[test] 启用测试时增强 —— {describe_tta(cfg.train.tta_hflip)}")
    test_metrics = evaluate(
        trainer.model, test_loader, criterion, device, amp=trainer.amp, topk=(1, 5),
        tta=use_tta, tta_hflip=bool(cfg.train.tta_hflip),
    )
    logger.info(
        f"[test] loss={test_metrics['loss']:.4f} | "
        f"Top-1 = {test_metrics['top1']:.2f}% | Top-5 = {test_metrics['top5']:.2f}% | "
        f"样本数 = {int(test_metrics['num_samples'])} | TTA = {use_tta}"
    )

    # ------------------------------------------------------------------ #
    # 8. 逐类准确率 / 混淆矩阵 / 图表
    # ------------------------------------------------------------------ #
    # 与上面的 evaluate 保持完全相同的推理口径（是否 TTA 必须一致），
    # 否则混淆矩阵/逐类准确率会与总体 Top-1 对不上
    y_true, y_pred = collect_predictions(
        trainer.model, test_loader, device, amp=trainer.amp,
        tta=use_tta, tta_hflip=bool(cfg.train.tta_hflip),
    )
    analysis = analyze_predictions(
        y_true, y_pred, class_names,
        logger=logger,
        save_path=os.path.join(cfg.result_dir, "per_class_metrics.json"),
    )

    plot_confusion_matrix(
        np.asarray(analysis["confusion_matrix"]),
        class_names,
        save_path=os.path.join(cfg.fig_dir, "confusion_matrix.png"),
        normalize=True,
    )
    plot_per_class_accuracy(
        [analysis["per_class_accuracy"][name] for name in class_names],
        class_names,
        save_path=os.path.join(cfg.fig_dir, "per_class_accuracy.png"),
    )

    # ------------------------------------------------------------------ #
    # 9. 汇总并落盘
    # ------------------------------------------------------------------ #
    total_seconds = time.time() - wall_start
    summary = {
        "experiment": cfg.exp_name,
        "algorithm": "ConvNeXt (A ConvNet for the 2020s, CVPR 2022; arXiv:2201.03545)",
        "architecture": {
            "preset": cfg.model.name,
            "stem_mode": cfg.model.stem_mode,
            "drop_path_rate": cfg.model.drop_path_rate,
            "depths": list(cfg.model.depths) if cfg.model.depths else None,
            "dims": list(cfg.model.dims) if cfg.model.dims else None,
            "num_params": int(total_params),
        },
        "dataset": {
            "name": cfg.data.dataset,
            "is_synthetic": str(cfg.data.dataset).strip().lower() == "synthetic",
            "split": f"train {len(train_loader.dataset)} / val {len(val_loader.dataset)} / "
                     f"test {len(test_loader.dataset)}",
            # 只有"真实 CIFAR-10 + 全量数据"才算全量实验
            "is_full_scale": bool(
                str(cfg.data.dataset).strip().lower() in ("cifar10", "cifar-10", "cifar")
                and len(train_loader.dataset) == 45000
                and len(test_loader.dataset) == 10000
            ),
            "train_subset_size": cfg.data.train_subset_size,
            "test_subset_size": cfg.data.test_subset_size,
            "val_ratio": cfg.data.val_ratio,
            "data_root": os.path.abspath(cfg.data.data_root),
        },
        "device": describe_device(device),
        "gpu_total_memory_gb": round(gpu_total_memory_gb(device), 2),
        "batch_size_requested": int(cfg.train.batch_size),
        "batch_size_actual": int(actual_batch_size),
        "grad_accum_steps": int(accum_steps),
        "effective_batch_size": int(actual_batch_size * accum_steps),
        "epochs": int(cfg.train.epochs),
        "epochs_completed": len(history.get("epoch", [])),
        # 若不为 None，说明本轮被 --max-steps-per-epoch 限制，指标只代表该 step 子集
        "max_steps_per_epoch": cfg.train.max_steps_per_epoch,
        "best_val_acc": round(float(trainer.best_val_acc), 4),
        "best_val_epoch": int(trainer.best_epoch),
        "test_metrics": {
            "loss": round(float(test_metrics["loss"]), 6),
            "top1_acc": round(float(test_metrics["top1"]), 4),
            "top5_acc": round(float(test_metrics["top5"]), 4),
            "num_samples": int(test_metrics["num_samples"]),
        },
        "evaluation": {
            "tta": use_tta,
            "tta_views": (["identity", "hflip"] if (use_tta and cfg.train.tta_hflip) else ["identity"]),
            "note": "loss 由不做 TTA 的原始前向计算；Top-1/Top-5 使用 TTA 平均 softmax 概率",
        },
        "per_class_accuracy": {k: round(float(v) * 100.0, 4)
                               for k, v in analysis["per_class_accuracy"].items()},
        "macro_accuracy": round(float(analysis["macro_accuracy"]), 4),
        "total_training_time": human_time(total_seconds),
        "seed": int(cfg.train.seed),
        # 混合精度可追溯性：fp16 的 GradScaler scale 若崩塌到 <1，
        # 会导致部分参数梯度下溢为 0（曾造成"loss 不降、精度停在 25%"的假收敛）。
        # 这里记录最小 scale 与是否触发过自动降级，作为结果可信度的判定依据。
        "amp": {
            "enabled": bool(trainer.amp),
            "dtype": (str(trainer.amp_dtype).replace("torch.", "") if trainer.amp_dtype else None),
            "min_scale": trainer.amp_min_scale,
            "fell_back_to_fp32": bool(trainer.amp_fell_back),
        },
    }
    save_json(summary, os.path.join(cfg.result_dir, "test_metrics.json"))
    # 同时导出一行式指标汇总表，便于把多次实验直接拼接成对比/消融表格
    csv_path = write_metrics_csv(summary, os.path.join(cfg.result_dir, "metrics.csv"))
    logger.info(f"[metrics] 指标汇总表已保存：{os.path.abspath(csv_path)}")

    logger.info("=" * 96)
    logger.info(f"全部完成！总耗时 {human_time(total_seconds)}")
    logger.info(f"最终测试集 Top-1 准确率：{test_metrics['top1']:.2f}%")
    logger.info(f"本次实验产物目录：{os.path.abspath(cfg.run_root)}")
    logger.info(f"结果文件：{os.path.abspath(cfg.result_dir)}")
    logger.info(f"图表文件：{os.path.abspath(cfg.fig_dir)}")
    logger.info("=" * 96)


if __name__ == "__main__":
    main()
