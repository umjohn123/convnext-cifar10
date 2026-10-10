# -*- coding: utf-8 -*-
"""
ConvNeXt + CIFAR-10 独立评估入口脚本。

用于在不重新训练的前提下，用已保存的检查点对**测试集**做完整评估：

    !python evaluate.py --ckpt ./outputs/models/best.pt
    !python evaluate.py --ckpt ../05_results/<实验标签>/models/best.pt

若省略 ``--ckpt``，默认使用 ``<output-dir>/<run-tag>/models/best.pt``；
评估产物会写入该检查点所属实验自己的目录，不与其它实验交叉。

输出内容
--------
* 测试集损失、Top-1 / Top-5 准确率；
* 逐类准确率（按类别打印）与宏平均；
* 混淆矩阵图、逐类准确率条形图；
* ``<实验目录>/metrics/eval_metrics.json`` 等结构化结果文件。

设计要点
--------
1. **自动恢复训练配置**：检查点中保存了完整的训练配置（``config`` 字段），
   本脚本会以它为基准重建模型结构，因此**无需手动指定** ``--model`` / ``--stem-mode``
   等结构参数即可正确加载权重（若显式传入命令行参数，则以命令行优先）。
2. **划分一致性**：使用与训练**完全相同的随机种子**划分验证集，
   测试集始终是官方划分的 10000 张图像，不存在数据泄漏或划分漂移。
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from configs import Config, parse_args
from src.data import build_dataloaders
from src.engine.evaluator import analyze_predictions, collect_predictions, evaluate
from src.models.convnext import build_convnext, convnext_model_summary
from src.utils.device import describe_device, get_device
from src.utils.logger import get_logger
from src.utils.misc import count_parameters, format_params, save_json
from src.utils.seed import set_seed
from src.utils.tta import describe_tta
from src.utils.visualize import plot_confusion_matrix, plot_per_class_accuracy

_IMAGE_SIZE = 32


def parse_eval_args(argv: Optional[List[str]] = None) -> Tuple[Config, str, List[str]]:
    """解析评估脚本参数。

    先提取评估专用的 ``--ckpt``，其余参数原样交给通用配置解析器。

    返回
    ----
    (cfg, ckpt_path, remaining_args)
        ``cfg`` 为仅含命令行覆盖项的配置；``remaining_args`` 用于在读取检查点后
        以检查点配置为基准做二次解析。
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ckpt", type=str, default=None, help="待评估的检查点路径")
    known, remaining = parser.parse_known_args(argv)

    cfg = parse_args(remaining)
    ckpt_path = known.ckpt or os.path.join(cfg.ckpt_dir, "best.pt")
    return cfg, ckpt_path, list(remaining)


def main() -> None:
    """评估主流程。"""
    cfg, ckpt_path, remaining_args = parse_eval_args()

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"找不到检查点 {os.path.abspath(ckpt_path)}，请先运行 train.py，"
            f"或用 --ckpt 指定正确路径。"
        )

    # ------------------------------------------------------------------ #
    # 1. 读取检查点，并用其中保存的训练配置作为基准重建配置
    # ------------------------------------------------------------------ #
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    restored_from_ckpt = False
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        base_cfg = Config.from_dict(checkpoint["config"])
        # 命令行的显式参数优先级更高（parse_args 只覆盖被显式传入的项）
        cfg = parse_args(remaining_args, base=base_cfg)
        restored_from_ckpt = True

    cfg.make_dirs()
    logger = get_logger(name="convnext_eval", log_file=os.path.join(cfg.log_dir, "evaluate.log"))

    logger.info("=" * 100)
    logger.info(f"[evaluate] 检查点：{os.path.abspath(ckpt_path)}")
    if restored_from_ckpt:
        logger.info(
            f"[evaluate] 已从检查点恢复训练时的模型配置：preset={cfg.model.name} | "
            f"stem_mode={cfg.model.stem_mode} | drop_path={cfg.model.drop_path_rate}"
        )

    # ------------------------------------------------------------------ #
    # 2. 设备与随机种子（与训练保持一致，保证数据划分不变）
    # ------------------------------------------------------------------ #
    set_seed(seed=int(cfg.train.seed), deterministic=bool(cfg.train.deterministic))
    device = get_device(cfg.device)
    logger.info(f"[device] {describe_device(device)}")

    # ------------------------------------------------------------------ #
    # 3. 数据（评估只需要测试集，这里一并构建以便复用同一套划分逻辑）
    # ------------------------------------------------------------------ #
    eval_batch_size = max(64, int(cfg.train.batch_size) * 2)
    _, _, test_loader, class_names = build_dataloaders(
        cfg, batch_size=int(cfg.train.batch_size), eval_batch_size=eval_batch_size, logger=logger
    )

    # ------------------------------------------------------------------ #
    # 4. 模型与权重
    # ------------------------------------------------------------------ #
    model = build_convnext(cfg.model, num_classes=cfg.data.num_classes)
    total_params, _ = count_parameters(model)
    for line in convnext_model_summary(cfg.model, num_classes=cfg.data.num_classes,
                                       input_size=_IMAGE_SIZE).split("\n"):
        logger.info(f"    {line}")
    logger.info(f"[model] 总参数量 = {format_params(total_params)}")
    model = model.to(device)

    state_dict = checkpoint.get("model_state", checkpoint)  # 兼容只保存 state_dict 的简单情况
    if isinstance(checkpoint, dict) and "class_names" in checkpoint and checkpoint["class_names"]:
        class_names = list(checkpoint["class_names"])
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as err:
        raise RuntimeError(
            f"权重与模型结构不匹配。\n{err}\n"
            f"提示：本脚本会自动从检查点恢复结构；若仍报错，请显式指定 "
            f"--model / --stem-mode（当前为 {cfg.model.name} / {cfg.model.stem_mode}）。"
        ) from err
    if missing or unexpected:
        logger.warning(f"权重加载存在差异 | 缺失键 {len(missing)} 个 | 多余键 {len(unexpected)} 个")
    logger.info(f"[evaluate] 已载入权重（训练时 epoch = {checkpoint.get('epoch', 'N/A') if isinstance(checkpoint, dict) else 'N/A'}）")

    # ------------------------------------------------------------------ #
    # 5. 评估
    # ------------------------------------------------------------------ #
    amp = bool(cfg.train.amp) and device.type == "cuda"
    use_tta = bool(cfg.train.tta)
    if use_tta:
        logger.info(f"[test] 启用测试时增强 —— {describe_tta(cfg.train.tta_hflip)}")
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(cfg.train.label_smoothing))
    test_metrics = evaluate(
        model, test_loader, criterion, device, amp=amp, topk=(1, 5),
        tta=use_tta, tta_hflip=bool(cfg.train.tta_hflip),
    )
    logger.info(
        f"[test] loss={test_metrics['loss']:.4f} | "
        f"Top-1 = {test_metrics['top1']:.2f}% | Top-5 = {test_metrics['top5']:.2f}% | "
        f"样本数 = {int(test_metrics['num_samples'])} | TTA = {use_tta}"
    )

    # ------------------------------------------------------------------ #
    # 6. 逐类指标与混淆矩阵
    # ------------------------------------------------------------------ #
    # 与上面的 evaluate 保持完全相同的推理口径（是否 TTA 必须一致）
    y_true, y_pred = collect_predictions(
        model, test_loader, device, amp=amp,
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
    # 7. 落盘
    # ------------------------------------------------------------------ #
    summary = {
        "checkpoint": os.path.abspath(ckpt_path),
        "device": describe_device(device),
        "batch_size": int(eval_batch_size),
        "model": {
            "preset": cfg.model.name,
            "stem_mode": cfg.model.stem_mode,
            "num_params": int(total_params),
        },
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
        "macro_accuracy": round(float(analysis["macro_accuracy"]), 4),
        "per_class_accuracy": {k: round(float(v) * 100.0, 4)
                               for k, v in analysis["per_class_accuracy"].items()},
        "confusion_matrix": analysis["confusion_matrix"],
        "seed": int(cfg.train.seed),
    }
    out_path = os.path.join(cfg.result_dir, "eval_metrics.json")
    save_json(summary, out_path)
    logger.info(f"[evaluate] 结果已保存：{os.path.abspath(out_path)}")
    logger.info(f"[evaluate] 图表已保存：{os.path.abspath(cfg.fig_dir)}")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
