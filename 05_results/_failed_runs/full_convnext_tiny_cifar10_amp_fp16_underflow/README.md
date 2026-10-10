# 失败运行归档：`full_convnext_tiny_cifar10`（fp16 AMP 梯度下溢 → 骨干网络冻结）

> ⚠️ 本目录是一次**训练未收敛（假收敛）**的运行记录，**不可作为报告的主结果**。
> 归档目的是保留完整的失败证据链，供报告"问题定位与解决"章节与后续实验对照使用。
> 修复后的正式运行请使用**新的实验标签**（例如 `full_convnext_tiny_cifar10`（重跑）或 `full_tiny_fp32`），
> 并写入 `05_results/<新标签>/`，与本目录互不覆盖。

---

## 1. 运行元信息

| 项目 | 值 |
|---|---|
| 实验标签（`--run-tag`） | `full_convnext_tiny_cifar10` |
| 运行环境 | Google Colab / **Tesla T4（14.6 GB）** / CUDA / PyTorch |
| 命令 | `python train.py --run-tag full_convnext_tiny_cifar10 --epochs 60 --model tiny --batch-size 128 --stem-mode conv` |
| 精度路径 | **AMP = fp16 + GradScaler**（默认开启，本次未加 `--no-amp`） |
| 数据 | 真实 CIFAR-10，全量（train 45000 / val 5000 / test 10000，`is_full_scale=true`） |
| 训练时长 | 1 h 54 m（60/60 轮全部完成） |
| 测试集 Top-1 / Top-5 | **25.73 % / 78.35 %**（预期 93–95 %） |
| 最佳验证轮次 | **epoch 2（26.66 %）**，其后 58 轮从未超越 |
| 报错/异常 | **无**：`train.log` / `evaluate.log` 中无 error / traceback / warning / nan / inf |

**结论：流程正确、无报错，但模型没有真正学到特征 —— 属于"看似跑完、实则假收敛"的运行，结果无效。**

---

## 2. 现象证据

**逐轮指标**（`logs/history.csv`，节选）：

| epoch | train_loss | train_acc | val_acc |
|---|---|---|---|
| 1 | 2.2038 | 18.80 % | 24.84 % |
| 2 | 2.1479 | 22.44 % | **26.66 %（最优）** |
| 5 | 2.1516 | 21.30 % | 25.34 % |
| 10 | 2.1601 | 21.44 % | 24.60 % |
| 24 | 2.2725 | 14.64 % | 11.56 % |
| 40 | 2.2177 | 17.78 % | 19.04 % |
| 60 | 2.1846 | 20.37 % | 23.40 % |

* 交叉熵始终停在 **2.15 ~ 2.27**（10 类随机水平为 ln10 = 2.303），60 轮没有任何下降趋势；
* 曲线见 `figures/training_curves.png`：train/val 曲线全程水平抖动；
* 预测分布严重坍缩（`metrics/per_class_metrics.json` 的混淆矩阵列和）：**frog 2527 / ship 2026**，
  二者占全部预测的 45 %，而 `dog` 只有 653；
* 逐类准确率两极分化：`ship 61.1 %`、`frog 59.9 %`，而 `cat 5.0 %`、`bird 7.2 %`。

**独立复算（本地 CPU / 真实 CIFAR-10，用于排除数据问题）**：

| 复算对象 | 结果 | 与 Colab 报告对比 |
|---|---|---|
| `best.pt` 在测试集 10000 张 | Top-1 = **25.71 %**、Top-5 = 78.36 % | Colab 25.73 % / 78.35 % → **一致** |
| `best.pt` 在训练集前 4000 张（无增强） | Top-1 = **27.12 %** | 连训练集都拟合不了 → **严重欠拟合（非过拟合）** |
| `last.pt` 在测试集 10000 张 | Top-1 = 23.69 % | 第 60 轮反而更差 |

→ 本地数据与 Colab 数据等价，**排除"数据集损坏 / 标签错乱 / 划分漂移"**。

---

## 3. 根因证据链

### 3.1 直接证据：GradScaler 的动态 scale 崩塌

从检查点内保存的 `scaler_state` 读出（PyTorch 默认初值为 **65536**）：

| 检查点 | `scale` | `growth_tracker` | `growth_interval` |
|---|---|---|---|
| `best.pt`（epoch 2） | **0.015625** | 154 | 2000 |
| `last.pt`（epoch 60） | **0.0625** | 1258 | 2000 |

* `65536 → 0.015625` 相当于**连续 22 次减半**，即训练早期发生了 20 次以上 fp16 梯度溢出（inf/NaN）；
* 之后 scale 只能按 `growth_interval=2000` 缓慢回升：5200 步（约 2 次倍增）→ 0.0625，与观测完全吻合。

### 3.2 决定性证据：骨干参数**从未收到梯度**

`last.pt` 中 AdamW 的 `exp_avg_sq`（梯度平方的指数滑动平均，恒为 0 意味着**该参数在任何一步都没有非零梯度**）：

| 参数 | `sqrt(exp_avg_sq)`（≈ 梯度 RMS） | 判读 |
|---|---|---|
| `head.weight` | 4.26e-03 | 正常更新 |
| `downsample_layers.0.0.weight`（stem） | 2.96e-04 | 正常更新 |
| `norm.weight` / `norm.bias` | 9.45e-04 / 1.41e-03 | 正常更新 |
| `stages.2.*.gamma`（LayerScale） | 1e-05 ~ 6.7e-04 | 有更新（γ 的梯度不含 γ 因子） |
| `stages.0.*.pwconv1.weight` | **2.4e-14** | 几乎冻结 |
| `stages.1.*.pwconv1.weight` | **0.000e+00** | **全程精确为 0 → 权重一次都没更新过** |
| `stages.2.*.pwconv1.weight` | 1e-09 ~ 5e-07 | 极其微弱 |
| `stages.3.*.pwconv1.weight` | 2.7e-08 ~ 8.9e-08 | 极其微弱 |

**即：实际被训练的只有 stem + 分类头 + 末端 GroupNorm（外加部分 LayerScale γ），
等价于一个"3×3 卷积 + 全局池化 + 线性分类器"的浅层线性模型 —— 在 CIFAR-10 上正是 25 %~30 %。**

### 3.3 数值机制：真实梯度 × scale 后低于 fp16 可表示范围

用 `last.pt` 的权重在真实数据上做 **fp32 单步反向**（batch = 64，取梯度 RMS）：

| 参数 | fp32 梯度 RMS | ×scale(0.0156) | fp16 判定 |
|---|---|---|---|
| `head.weight` | 4.286e-02 | 6.7e-04 | 正常 |
| `downsample_layers.0.0.weight` | 3.044e-03 | 4.8e-05 | 正常 |
| `stages.3.2.pwconv1.weight` | 3.304e-05 | 5.2e-07 | 次正规区，精度极差 |
| `stages.0.0.pwconv1.weight` | 1.953e-06 | 3.1e-08 | **< 6e-8 → 取整为 0** |
| `stages.1.0.pwconv1.weight` | 1.809e-06 | 2.8e-08 | **< 6e-8 → 取整为 0** |

* fp16 的最小正规数 ≈ **6.10e-05**，最小次正规数 ≈ **5.96e-08**；GPU 上的 fp16 内核在次正规区
  普遍表现为 flush-to-zero，因此乘上小 scale 后的微小梯度会被直接抹成 0；
* 于是 `backward()` 得到"精确的 0 梯度" → AdamW 无任何更新方向 → 参数永久冻结（对应 3.2 的 `exp_avg_sq = 0`）。

### 3.4 为什么只有"浅层骨干"被冻死

本模型 LayerScale（γ）初值为 `1e-6`，且残差块内参数（dwconv / pwconv1 / pwconv2）的梯度**都带 γ 因子**：

* 残差主路（identity）的梯度不受 γ 影响 → **stem 的梯度健康（3e-3）**，所以 stem 被训练了；
* 越靠近输入，进入分支的梯度越小，再乘 `γ = 1e-6` 后彻底沉入 fp16 的不可表示区 → **浅层 Block 冻结**；
* γ 自身的梯度 `∂L/∂γ = ⟨grad_out, branch_out⟩` **不含 γ 因子**，因此 γ 反而能长大
  （stage 3 的 γ 从 1e-6 长到最大值 1.25），这也解释了"γ 很大但网络照样不学习"的矛盾现象。

### 3.5 反例对照（同一份代码，fp32 正常收敛）

| 实验 | 精度 | 数据 | 结果 |
|---|---|---|---|
| `mvp_atto_subset5k`（`05_results/mvp_atto_subset5k/`） | **fp32（CPU）** | 真实 CIFAR-10 子集 | loss 2.13 → 1.68，val 27 % → **45.05 %**，**正常收敛** |
| 本次运行 | **fp16 AMP（T4）** | 真实 CIFAR-10 全量 | loss 2.20 → 2.18，val 峰值 26.66 %，**假收敛** |

→ 问题定位在 **fp16 / GradScaler 这条数值路径**，与模型定义、数据划分、评估口径无关。

---

## 4. 已实施的修复（见 `03_code`）

| 位置 | 改动 |
|---|---|
| `src/utils/amp.py` | 新增 `resolve_amp_dtype()` / `bf16_natively_supported()` / `describe_amp()`：**默认优先 bfloat16**（阶码 8 位，与 fp32 同量程，无下溢），仅在不支持原生 bf16 的设备（T4 等）回退 fp16；`autocast_context()` 支持显式 `dtype` |
| `src/engine/trainer.py` | ① 每轮记录并打印 `amp_scale`；② `amp_scale` 跌破 `AMP_SCALE_FLOOR = 1.0` 时打印告警并**自动降级为 fp32 继续训练**（`_track_amp_scale()`，配置项 `train.amp_fallback`）；③ 启动时若只能用 fp16 则给出显式提示；④ 检查点内保存 `amp{enabled,dtype,min_scale,fell_back_to_fp32}` |
| `src/utils/csv_logger.py` | `history.csv` 增加 `amp_scale` 列；`metrics.csv` 增加 `amp_dtype` / `amp_min_scale` 列，便于事后判定一次运行是否可信 |
| `configs.py` | 新增 `train.amp_fallback`（默认 `True`）与 `--no-amp-fallback` 开关 |
| `train.py` | 结果汇总 `test_metrics.json` 内写入 `amp` 字段（含 `min_scale`、是否降级） |
| `notebooks/ConvNeXt_CIFAR10_Colab.ipynb` | 3-C 正式训练单元格默认加 **`--no-amp`**，并写明 T4 的 fp16 风险与 L4/A100 的 bf16 替代方案（由生成器 `build_colab_notebook.py` 重新生成，自检通过） |

**修复后的执行建议（T4）**：

```bash
# 先用 3~5 分钟验证 loss 会下降，再跑全量
!python train.py --run-tag verify_fp32 --epochs 3 --model tiny --batch-size 128 --stem-mode conv \
                 --no-amp --train-subset-size 5000 --test-subset-size 2000 --exp-name verify_fp32

# 正式全量（T4 上 fp32 约 3~5 小时；断线可用 --resume auto 续训）
!python train.py --run-tag full_convnext_tiny_cifar10 --epochs 60 --model tiny \
                 --batch-size 128 --stem-mode conv --no-amp
!python evaluate.py --ckpt outputs/full_convnext_tiny_cifar10/models/best.pt --tta
```

> 若把运行时换成 **L4 / A100**（原生支持 bf16），可去掉 `--no-amp`：程序会自动使用 bfloat16
> （阶码与 fp32 相同，不存在本次的下溢问题）；务必确认日志中 `AMP=True（autocast bfloat16 + GradScaler）`
> 且每轮 `amp_scale` 稳定在 5e4 量级。

---

## 5. 本目录归档范围

| 内容 | 说明 |
|---|---|
| `logs/train.log`、`logs/evaluate.log` | 完整训练/评估日志（无报错，含每轮 loss/acc） |
| `logs/history.csv`、`logs/history.json` | 逐轮指标（报告中"失败运行"的原始数据） |
| `metrics/config.json` | 本次运行的超参数快照（含 `amp=true`，可复现） |
| `metrics/test_metrics.json`、`metrics/metrics.csv` | 最终结果汇总（Top-1 = 25.73 %） |
| `metrics/per_class_metrics.json`、`metrics/eval_metrics.json` | 逐类准确率与 10×10 混淆矩阵 |
| `figures/*.png` | 4 张图表（曲线、混淆矩阵、逐类准确率、数据样例） |
| `models/*.pt` | **未解压**（best.pt / last.pt 各约 334 MB，保留在 `05_results/convnext_cifar10_outputs.zip` 内）；它们是本次诊断（`scaler_state`、`exp_avg_sq`）的原始依据，需要时可从 zip 单独取出 |

---

## 6. 修复后的复现验证（2026-10-10，本机 GPU）

按第 4 节的修复重新运行**同一套配方**（ConvNeXt-tiny、60 轮、真实全量 CIFAR-10、Mixup/CutMix、lr 1e-3、
batch 128、seed 42），本次在**本机 NVIDIA RTX 4060 Laptop（8 GB）** 上用**bfloat16 + GradScaler** 完成训练，
结果归档在 `../../full_convnext_tiny_cifar10/`：

| 对比项 | 本次失败运行（T4 + **fp16**） | 修复后重跑（RTX 4060 + **bf16**） |
|---|---|---|
| AMP 精度选择 | fp16（T4 不支持原生 bf16，工程当时默认 fp16） | **bfloat16**（Ampere 及以后自动优先；阶码与 fp32 相同） |
| GradScaler 动态 scale | **0.0156 / 0.0625**（默认 65536，连减 22 次） | 6.55e+04 → **6.71e+07**（单调健康增长） |
| 骨干参数是否收到梯度 | `stages.1.*.pwconv1.weight` 的 `exp_avg_sq` **恒为 0** | 全网正常更新 |
| 训练损失（60 轮） | 2.15 ~ 2.27，几乎不动（ln10 = 2.303） | 2.03 → **0.87** |
| 验证准确率 | 峰值 **26.66 %**（epoch 2） | 峰值 **93.50 %**（epoch 59） |
| 测试集 Top-1 | **25.73 %**（无效） | **92.66 %**（TTA **92.87 %**） |

**结论：根因判断与修复均得到验证** —— 同一份代码、同一套超参数，只要走出 fp16 的动态范围陷阱
（改用 bf16，或在只能 fp16 的设备上加 `--no-amp`），即可正常收敛到预期精度区间。

### 6.1 顺带发现并修复的另外两处工程问题（同一台本机）

| 问题 | 现象 | 修复 |
|---|---|---|
| 显存探测在 Windows/WDDM 下误判 | 探针把 batch 从 128 放大到 **512**（日志"峰值 10856 MB / 总显存 8 GB"），真实训练立刻 OOM 或 `CUDA driver error: device not ready` | 要求 `探针峰值 ≤ 可用显存 × 70 %`，并把 WDDM 驱动报错视为"放不下"；实测改为 **batch 128**（峰值 2883 MB / 可用 7000 MB） |
| DataLoader 多进程共享内存提交耗尽 | 训练约 2 小时后崩溃：`Couldn't open shared file mapping ... error code: 1455`（= 系统提交内存不足，非显存） | 长跑改用 `--num-workers 0`；崩溃后可用 `--resume auto` 原地续训（本次即如此完成 epoch 33–60） |

> 上述两项已写入 `03_code/README.md` 第 12.4 / 12.5 节，并在 `../../full_convnext_tiny_cifar10/logs/README.md` 中留有时间线证据。
