# models —— 模型

> 结果分类：**【模型】** ｜ 训练保存的检查点文件与参数说明
> 由 `03_code/train.py` → `src/engine/trainer.py` 在训练过程中自动保存（`torch.save`）。

---

## 1. 模型文件清单

| 文件名 | 内容 | 保存时机 | 用途 |
|---|---|---|---|
| `best.pt` | 验证集准确率**最优**的模型检查点（含完整训练状态） | 每当验证集 Top-1 准确率刷新历史最优时覆盖保存 | 最终测试评估、报告引用的"最优模型"、`evaluate.py` 的默认输入 |
| `last.pt` | **最后一轮**的模型检查点（含完整训练状态） | 每轮训练结束后覆盖保存 | 断点续训（`--resume`）；观察训练末期状态 |

> 两个文件都是**完整检查点**而非纯权重，因此既可以推理，也可以无缝续训。

---

## 2. 检查点内部结构

`torch.load(path, map_location=..., weights_only=False)` 返回一个 `dict`，键如下：

| 键 | 类型 | 含义 |
|---|---|---|
| `epoch` | `int` | 该检查点对应的轮次（0 起；显示时需 +1） |
| `model_state` | `dict` | 模型权重 `state_dict`（可直接 `load_state_dict`） |
| `optimizer_state` | `dict` | AdamW 优化器状态（动量、二阶矩、参数组） |
| `scheduler_state` | `dict` | 余弦退火调度器状态（用于精确续训） |
| `scaler_state` | `dict` / `None` | AMP `GradScaler` 状态；CPU 训练时为 `None` |
| `best_val_acc` | `float` | 截至该轮的历史最优验证准确率（百分数，0~100） |
| `best_epoch` | `int` | 取得历史最优的轮次（1 起） |
| `history` | `dict` | 完整训练历史（与 `../logs/history.json` 相同） |
| `config` | `dict` | **本次训练的全部超参数**（与 `../metrics/config.json` 相同） |
| `class_names` | `list[str]` | 类别名称顺序，保证预测索引与标签可正确对应 |

### 2.1 加载示例

```python
# 在 03_code 目录下执行（该目录含 configs.py 与 src/ 包）
import torch
from configs import Config
from src.models.convnext import build_convnext

CKPT = "../05_results/mvp_atto_subset5k/models/best.pt"
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)

print("轮次:", ckpt["epoch"] + 1, "| 最优验证准确率:", ckpt["best_val_acc"])
print("类别顺序:", ckpt["class_names"])

# 检查点内自带 config，可据此自动复原模型结构（03_code/evaluate.py 正是这样做的）
cfg = Config.from_dict(ckpt["config"])
model = build_convnext(cfg.model, num_classes=cfg.data.num_classes)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("参数量:", sum(p.numel() for p in model.parameters()))
```

> `03_code/evaluate.py` 正是利用 `config` 字段**自动恢复模型结构**，
> 因此评估时无需手动指定 `--model` / `--stem-mode` 等结构参数。

---

## 3. 本次存档模型的架构与参数

| 项目 | 值 |
|---|---|
| 算法 | **ConvNeXt**（*A ConvNet for the 2020s*, CVPR 2022, arXiv:2201.03545） |
| 预设规模 | `atto` |
| Stem 类型 | `conv`（3×3、stride=1，适配 32×32 输入） |
| 阶段通道数 | 40 / 80 / 160 / 320 |
| 阶段 Block 数 | 2 / 2 / 6 / 2 |
| 各阶段分辨率 | 32×32 → 16×16 → 8×8 → 4×4 |
| Stochastic Depth | `drop_path_rate = 0.1`（各 Block 由 0 线性递增到 0.1） |
| Layer Scale 初值 | `1e-6` |
| 分类头 | 全局平均池化 → `Linear(320 → 10)` |
| **总参数量** | **3,376,890（3.38 M）** |
| 可训练参数量 | 3,376,890（全参数训练，无冻结层） |

**关键结构组件**（完整保留 ConvNeXt 论文设计）：
7×7 深度卷积 · 倒置瓶颈（1×1 升维 4 倍 → 深度卷积 → 1×1 降维）· LayerNorm 替代 BatchNorm ·
GELU 替代 ReLU（每块仅 1 个激活）· 独立下采样层（LayerNorm + 2×2 stride-2 卷积）·
Layer Scale · Stochastic Depth。

---

## 4. 训练超参数（与检查点内 `config` 一致）

| 超参数 | 值 |
|---|---|
| 优化器 | AdamW（1 维参数豁免权重衰减） |
| 基础学习率 / 最小学习率 | 2.0e-3 / 1e-5 |
| 权重衰减 | 0.05 |
| 调度 | 线性 warmup 2 轮 + 余弦退火（共 10 轮） |
| 标签平滑 | 0.1 |
| Mixup / CutMix | 关闭（`--no-mixup`，小规模 MVP 实验） |
| 数据增强 | RandomCrop(32, padding=4, reflect) + RandomHorizontalFlip + RandomErasing(p=0.25) |
| 批大小 | 128（等效 batch = 128，无梯度累积） |
| 随机种子 | 42 |
| 训练设备 | CPU（详见 `../logs/train.log` 的 `[device]` 行） |

---

## 5. 与其它部分的关系

| 关联 | 说明 |
|---|---|
| 模型 ← 数据 | 训练使用 `../../../04_data/cifar10`（划分索引见 `../../../04_data/splits/split_info.json`） |
| 模型 ← 日志 | `../logs/train.log` 记录了每次保存 `best.pt` 的时刻 |
| 模型 → 指标 | `../metrics/test_metrics.json` 由 `best.pt` 在测试集上评估得到 |
| 模型 → 图表 | `../figures/confusion_matrix.png`、`per_class_accuracy.png` 均由 `best.pt` 的预测生成 |
| 模型 → 复现 | 检查点自带 `config`，`evaluate.py` 可据此一键复现评估 |
