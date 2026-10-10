# AIassign2 候选算法筛选：5 个 2022–2026 年 Deep CNN / Deep Autoencoder 算法

| 项目 | 内容 |
|---|---|
| 文档名称 | AIassign2 候选算法筛选清单 |
| 作业主题 | **AIassign2**（作业文件 `AIassign2.pdf`，正文标题 **AI Assignment #1**） |
| 对应任务 | 任务拆解 **T3**（AI 检索算法与记录）/ 执行计划 **I0-1**（算法选题） |
| 文档版本 | v1.0 |
| 编制日期 | 2026-10-09 |
| 归档位置 | `02_ai_chats/algorithm_candidates.md` |
| 应用方向 | 计算机视觉与模式识别（图像分类、检测、分割、自监督表示学习） |

---

## 1. 筛选标准与判定方法

### 1.1 入选标准（三项均为必要条件）
| 编号 | 标准 | 判定方式 |
|---|---|---|
| C-1 | **类别合规**：属 Deep CNN 或 Deep Autoencoder | 核对网络主干与训练范式 |
| C-2 | **年份合规**：2022–2026 年间提出 | 核对 arXiv 首发时间与正式发表年份 |
| C-3 | **正式发表**：有公开正式论文 | 核对会议/期刊或开放获取版本 |
| C-4 | **开源可访问**：有公开可访问的官方代码仓库 | 核对官方 GitHub 链接 |
| C-5 | **Colab 可运行**：可在 Google Colab 免费环境直接运行 | 评估依赖复杂度、算力需求、运行时长 |

### 1.2 判定说明
- 年份以 **arXiv 首次公开时间** 与 **正式发表年份** 双向核对；若两者不一致，以正式发表年份为准并明确标注。
- Colab 可运行性按"免费版（T4 GPU / 约 12–16 GB 内存、单次会话时长受限）"评估。

---

## 2. 候选算法总览

| # | 算法名称 | 类别 | 发表 | 一句话定位 | Colab 友好度 |
|:--:|---|---|---|---|:--:|
| 1 | **ConvNeXt** | Deep CNN | CVPR 2022 | 现代化纯卷积主干，对标 Transformer | ★★★★★ |
| 2 | **ConvNeXt V2** | Deep CNN + 掩码自编码 | CVPR 2023 | FCMAE 自监督预训练 + GRN 层 | ★★★★☆ |
| 3 | **Masked Autoencoders (MAE)** | Deep Autoencoder | CVPR 2022 | 高掩码率自监督表示学习范式 | ★★★★☆ |
| 4 | **FasterNet** | Deep CNN | CVPR 2023 | PConv 提升实际推理速度 | ★★★★★ |
| 5 | **RepViT** | Deep CNN | CVPR 2024 | 从 ViT 视角重构轻量移动 CNN | ★★★★★ |

---

## 3. 各算法详解

### 3.1 ConvNeXt

| 项目 | 内容 |
|---|---|
| **算法名称** | ConvNeXt |
| **论文标题** | *A ConvNet for the 2020s* |
| **作者** | Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie |
| **发表年份** | **2022**（CVPR 2022；arXiv 首发 2022-01） |
| **论文链接** | https://arxiv.org/abs/2201.03545 ｜ https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html |
| **开源代码** | https://github.com/facebookresearch/ConvNeXt |

**核心思想**：以 ResNet 为起点，逐步"现代化"其设计（大核深度卷积、倒置瓶颈、LayerNorm、GELU、更少激活层等），最终得到一个**完全由标准卷积模块构建的纯 ConvNet**，在准确率与可扩展性上对标 Swin Transformer，证明卷积网络在 2020 年代仍具竞争力。

**优点**
- 结构清晰、实现简洁，全部为标准卷积算子，无自定义 CUDA 扩展。
- 已集成进 `timm`，一行代码即可调用预训练权重。
- 训练与微调流程成熟，复现成本低。

**缺点**
- 大核深度卷积在部分推理后端上速度优势不明显。
- 从头训练大模型需大量算力与 ImageNet 级数据。

**推荐数据集**：ImageNet-1K（分类，主力）；轻量验证可用 CIFAR-10 / CIFAR-100。

**Colab 可运行性**
- 依赖：`torch`、`torchvision`、`timm`（`pip install timm`），无自定义算子。
- 算力/时长：微调 `convnext_tiny` 于 CIFAR-10 或 ImageNet 子集，单次数十分钟内可完成；从零训练完整 ImageNet 不可行（需裁剪规模）。

---

### 3.2 ConvNeXt V2

| 项目 | 内容 |
|---|---|
| **算法名称** | ConvNeXt V2 |
| **论文标题** | *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders* |
| **作者** | Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie |
| **发表年份** | **2023**（CVPR 2023；arXiv 首发 2023-01） |
| **论文链接** | https://arxiv.org/abs/2301.00808 ｜ https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html |
| **开源代码** | https://github.com/facebookresearch/ConvNeXt-V2 |

**核心思想**：将**全卷积掩码自编码器（FCMAE）** 预训练框架与网络架构**协同设计**，并引入 **GRN（Global Response Normalization）** 层以增强通道间特征竞争，从而在纯卷积结构上复现掩码自监督预训练的良好扩展性。

**优点**
- 同时具备 CNN 主干的效率与自编码器预训练的表示能力，兼具两类特征。
- 官方实现为纯 PyTorch，模块化清晰；主干亦收录于 `timm`。
- 在分类、检测、分割多任务上均有公开结果。

**缺点**
- FCMAE 预训练阶段显存与时长需求高于纯监督微调。
- 完整预训练需 ImageNet 级数据，Colab 上只能做小规模复现。

**推荐数据集**：ImageNet-1K（预训练/微调）；小规模验证用 CIFAR-10/100 或 Tiny-ImageNet。

**Colab 可运行性**
- 依赖：`torch`、`torchvision`（可选 `timm`），官方仓库纯 PyTorch，无自定义扩展。
- 算力/时长：以预训练权重做下游微调在 Colab 可行；从零跑 FCMAE 预训练须大幅缩小模型与数据规模。

---

### 3.3 Masked Autoencoders (MAE)

| 项目 | 内容 |
|---|---|
| **算法名称** | Masked Autoencoders (MAE) |
| **论文标题** | *Masked Autoencoders Are Scalable Vision Learners* |
| **作者** | Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick |
| **发表年份** | **2022**（CVPR 2022 正式发表；arXiv 首发 2021-11） |
| **论文链接** | https://arxiv.org/abs/2111.06377 ｜ https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html |
| **开源代码** | https://github.com/facebookresearch/mae |

**核心思想**：随机掩蔽图像的大部分图块（如 75%），仅将可见图块送入 **非对称编码器-解码器**（编码器只处理可见块，轻量解码器重建被掩蔽像素），以像素重建为自监督目标学习可迁移的视觉表示。

**优点**
- 范式简洁优雅，是**深度自编码器**在视觉领域的代表性工作，与作业类别要求高度契合。
- 官方实现为纯 PyTorch，代码结构清晰、易于裁改。
- 预训练权重公开，下游微调流程成熟。

**缺点**
- 高掩码率下的预训练计算量大，完整 ImageNet 预训练需大规模算力。
- 原生面向 ViT 主干，直接用于小数据集时收益有限。

**推荐数据集**：ImageNet-1K（预训练 + 微调）；Colab 场景建议 CIFAR-10/100 或 Tiny-ImageNet 做小规模预训练/微调。

**Colab 可运行性**
- 依赖：`torch`、`torchvision`、`timm`，无自定义 CUDA 扩展。
- 算力/时长：使用官方预训练权重做下游微调可行；从零预训练须显著缩小规模与轮次。

> **年份合规提示**：MAE 的 arXiv 预印本首发于 2021-11，**正式发表于 CVPR 2022**。按"正式发表年份"计为 2022，符合 2022–2026 区间；若采用严格"首发年份"口径，建议改用附录中的替代算法。

---

### 3.4 FasterNet

| 项目 | 内容 |
|---|---|
| **算法名称** | FasterNet |
| **论文标题** | *Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks* |
| **作者** | Jierun Chen, Shiu-hong Kao, Hao He, Weipeng Zhuo, Song Wen, Chul-Ho Lee, S.-H. Gary Chan |
| **发表年份** | **2023**（CVPR 2023；arXiv 首发 2023-03） |
| **论文链接** | https://arxiv.org/abs/2303.03667 ｜ https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html |
| **开源代码** | https://github.com/JierunChen/FasterNet |

**核心思想**：指出现有网络"低 FLOPs 并不等于低延迟"，根源是频繁内存访问导致 **FLOPS（每秒浮点运算数）偏低**；为此提出 **PConv（部分卷积）**，仅对部分通道做卷积以减少冗余计算与访存，并据此构建 **FasterNet** 系列。

**优点**
- 依赖极简、模型轻量，**Colab 训练与推理速度表现优秀**。
- 提供从小型（T0）到大型的完整模型族，便于按算力选型。
- 纯 PyTorch 官方实现，含分类训练代码。

**缺点**
- 面向效率优化，绝对精度不追求 SOTA。
- 论文核心卖点是延迟而非精度，需搭配延迟测量才能体现价值。

**推荐数据集**：ImageNet-1K（分类）；目标检测用 COCO；轻量验证用 CIFAR-10。

**Colab 可运行性**
- 依赖：`torch`、`torchvision`（可选 PyTorch Lightning），无自定义算子。
- 算力/时长：`FasterNet-T0/T1` 在 Colab 上训练/微调速度快，适合快速迭代。

---

### 3.5 RepViT

| 项目 | 内容 |
|---|---|
| **算法名称** | RepViT |
| **论文标题** | *RepViT: Revisiting Mobile CNN From ViT Perspective* |
| **作者** | Ao Wang, Hui Chen, Zijia Lin, Jungong Han, Guiguang Ding |
| **发表年份** | **2024**（CVPR 2024；arXiv 首发 2023-07） |
| **论文链接** | https://arxiv.org/abs/2307.09283 ｜ https://openaccess.thecvf.com/content/CVPR2024/html/Wang_RepViT_Revisiting_Mobile_CNN_From_ViT_Perspective_CVPR_2024_paper.html |
| **开源代码** | https://github.com/THU-MIG/RepViT |

**核心思想**：以轻量 CNN（MobileNetV3）为起点，**从 ViT 的架构视角逐步改造**（引入结构重参数化、令牌混合与通道混合分离、更少激活层等），在保持移动端友好性的同时获得接近 ViT 的表达能力。

**优点**
- 极轻量、移动端友好，**Colab 上训练与推理开销最小**。
- 官方实现为纯 PyTorch，覆盖分类、检测、分割任务。
- 年份最新（CVPR 2024），满足"近期算法"要求。

**缺点**
- 定位为轻量模型，精度上限受参数量限制。
- 部分增益依赖结构重参数化，训练/推理代码需配套实现。

**推荐数据集**：ImageNet-1K（分类）；COCO（检测）；ADE20K（分割）；轻量验证用 CIFAR-10。

**Colab 可运行性**
- 依赖：`torch`、`torchvision`（检测/分割任务另需相应框架），无自定义 CUDA 扩展。
- 算力/时长：模型轻量，Colab 免费版即可完成微调与推理实验。

---

## 4. Colab 可运行性评估汇总

| 算法 | 依赖复杂度 | 自定义算子 | 算力需求 | Colab 时长风险 | 综合 |
|---|:--:|:--:|:--:|:--:|:--:|
| ConvNeXt | 低 | 无 | 中 | 低 | ★★★★★ |
| ConvNeXt V2 | 低 | 无 | 中–高（预训练阶段） | 中 | ★★★★☆ |
| MAE | 低 | 无 | 中–高（预训练阶段） | 中 | ★★★★☆ |
| FasterNet | 低 | 无 | 低 | 低 | ★★★★★ |
| RepViT | 低 | 无 | 低 | 低 | ★★★★★ |

> 说明：五个算法的官方实现均为**纯 PyTorch**，不含需编译的自定义 CUDA 扩展，因此在 Google Colab 上均可直接安装运行。风险主要来自"从头预训练"的规模，而非依赖问题——建议以**预训练权重 + 下游微调**或**缩小数据/模型规模**的方式在 Colab 上推进。

---

## 5. 筛选结论与推荐

### 5.1 标准符合性核对
| 算法 | C-1 类别 | C-2 年份 | C-3 正式论文 | C-4 开源 | C-5 Colab |
|---|:--:|:--:|:--:|:--:|:--:|
| ConvNeXt | ✔ Deep CNN | ✔ 2022 | ✔ CVPR 2022 | ✔ | ✔ |
| ConvNeXt V2 | ✔ CNN + AE | ✔ 2023 | ✔ CVPR 2023 | ✔ | ✔ |
| MAE | ✔ Deep AE | △ 见提示 | ✔ CVPR 2022 | ✔ | ✔ |
| FasterNet | ✔ Deep CNN | ✔ 2023 | ✔ CVPR 2023 | ✔ | ✔ |
| RepViT | ✔ Deep CNN | ✔ 2024 | ✔ CVPR 2024 | ✔ | ✔ |

### 5.2 推荐优先级
1. **首选：ConvNeXt（2022）**——类别（Deep CNN）与年份（2022）均严格合规，实现最简洁、Colab 友好度最高，最适合作为 I0/I1 的主算法。
2. **次选：FasterNet（2023）/ RepViT（2024）**——同样严格合规且更轻量，适合算力受限或需要快速迭代的场景。
3. **备选：ConvNeXt V2（2023）**——若希望同时体现"卷积 + 自编码预训练"两类特征，可选此项。
4. **备选：MAE（2022）**——最能体现"Deep Autoencoder"类别，但需注意其 arXiv 首发年份为 2021（正式发表 2022）。

---

## 6. 附录：其他备选算法（未列入主表）

| 算法名称 | 类别 | 发表 | 论文 | 开源代码 | 备注 |
|---|---|---|---|---|---|
| Visual Attention Network (VAN) | Deep CNN（大核注意力） | 2022 | https://arxiv.org/abs/2202.09741 | https://github.com/Visual-Attention-Network/VAN-Classification | 纯 PyTorch，Colab 友好 |
| ConvMAE | Deep Autoencoder | 2022（NeurIPS） | https://arxiv.org/abs/2205.03892 | https://github.com/Alpha-VL/ConvMAE | 严格 2022 首发，可替代 MAE |

---

## 7. 关联文档

- 任务描述（约束与验收标准）：`01_requirements/task_description.md`
- 执行计划（I0-1 算法选题）：`02_ai_chats/ai_collaboration_plan.md`
- 任务拆解（T1–T10）：`01_requirements/task_breakdown.md`
