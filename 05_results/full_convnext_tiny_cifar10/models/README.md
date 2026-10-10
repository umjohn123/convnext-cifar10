# models —— 模型检查点说明（`full_convnext_tiny_cifar10`）

> **归属任务**：T6（实验运行与结果收集）→【模型】产物
> 两个文件都是**完整训练状态检查点**（不仅权重），因此既能用于评估，也能用于**断点续训**。

## 1. 文件清单

| 文件 | 大小 | 对应轮次 | 说明 |
|---|---|---|---|
| `best.pt` | 334,097,589 B（约 334 MB） | **epoch 59**（验证集最优，val_top1 = **93.50 %**） | 报告与图表所用的模型：`test_metrics.json`、`per_class_metrics.json`、图 2、图 3 均由它评估得到 |
| `last.pt` | 334,097,653 B（约 334 MB） | **epoch 60**（最后一轮，val_top1 = 93.42 %） | 用于 `--resume auto` 原地续训 |

## 2. 检查点结构（两者字段一致）

```python
{
  "epoch":            int,    # 该检查点对应的轮次，0 基（58 → 第 59 轮；59 → 第 60 轮）
  "model_state":      dict,   # 模型权重（ConvNeXt-tiny，27,825,802 参数）
  "optimizer_state":  dict,   # AdamW 状态（含 exp_avg / exp_avg_sq），续训必需
  "scheduler_state":  dict,   # warmup + cosine 调度器状态（保证续训后学习率连续）
  "scaler_state":     dict,   # GradScaler 状态（scale / growth_tracker），bf16 路径的稳定性证据
  "best_val_acc":     float,  # 历史最优验证准确率（本实验 93.50）
  "best_epoch":       int,    # 最优轮次（1 基）
  "history":          dict,   # 逐轮指标（epoch / train_loss / train_acc / val_loss / val_acc / lr / epoch_time / amp_scale）
  "config":           dict,   # 本次运行的完整超参数快照（与 metrics/config.json 同源）
  "class_names":      list,   # 10 个类别名（与训练时一致，评估脚本据此保证类别顺序不变）
  "amp":              dict,   # 混合精度元信息：{enabled, dtype, min_scale, fell_back_to_fp32}
}
```

本实验的 `amp` 字段实际值为：

```json
{"enabled": true, "dtype": "bfloat16", "min_scale": 2097152.0, "fell_back_to_fp32": false}
```

> `min_scale = 2.10e+06` 说明 bf16 路径**从未出现梯度下溢**（对照：fp16 事故运行的最小 scale 为 **0.0156**，
> 见 `../../_failed_runs/full_convnext_tiny_cifar10_amp_fp16_underflow/`）。

## 3. 架构与超参数（写入检查点，可复现）

| 项目 | 值 |
|---|---|
| 模型 | ConvNeXt-tiny：`depths = (3,3,9,3)`，`dims = (96,192,384,768)` |
| Stem | `conv`（3×3、stride=1，适配 32×32 输入；四阶段分辨率 32 → 16 → 8 → 4） |
| 关键模块 | 倒置瓶颈（1×1 升维 4 倍 → 7×7 深度卷积 → 1×1 降维）、GELU、LayerNorm、Layer Scale(初值 1e-6)、DropPath(0.1)、独立下采样层 |
| 分类头 | 全局平均池化 → `Linear(768 → 10)`（`GroupNorm(1, C)` 末端归一化） |
| 参数量 | 27,825,802（27.83 M） |
| 优化器 / 调度 | AdamW(lr=1e-3, wd=0.05，1 维参数豁免权重衰减) + warmup 5 轮 + cosine → 1e-5 |
| 训练轮数 / 批大小 | 60 / 128（等效 batch 128） |
| 数据增强 | RandomCrop(padding=4, reflect) + RandomHorizontalFlip + RandomErasing(0.25) + Mixup/CutMix(prob 0.5) |
| 混合精度 | bfloat16 + GradScaler（CUDA 上自动选择；本机 RTX 4060 原生支持 bf16） |

## 4. 加载与使用示例

```python
import torch, sys
sys.path.insert(0, "../../03_code")
from configs import Config
from src.models.convnext import build_convnext

ck = torch.load("best.pt", map_location="cpu", weights_only=False)
print(ck["epoch"], ck["best_val_acc"], ck["best_epoch"], ck["amp"])

cfg = Config.from_dict(ck["config"])           # 用检查点里的配置重建模型结构（无需手写 --model）
model = build_convnext(cfg.model, num_classes=cfg.data.num_classes)
model.load_state_dict(ck["model_state"])
model.eval()
```

命令行等价用法：

```bash
# 评估（工程会自动从检查点恢复结构；--tta 启用测试时增强）
python evaluate.py --ckpt ../05_results/full_convnext_tiny_cifar10/models/best.pt --device cuda --tta

# 续训（模型 / 优化器 / 调度器 / 历史全部恢复）
python train.py --run-tag full_convnext_tiny_cifar10 --model tiny --epochs 60 --batch-size 128 \
                --device cuda --num-workers 0 --resume ../05_results/full_convnext_tiny_cifar10/models/last.pt \
                --data-root ../04_data/cifar10 --output-dir ../05_results
```

> ⚠️ 体积提示：单个检查点约 334 MB（含 AdamW 一阶/二阶动量，故远大于纯权重）。
> 若托管到 Git，建议忽略 `*.pt`（见 `../../README.md` 第 8 节的 `.gitignore` 建议）。
