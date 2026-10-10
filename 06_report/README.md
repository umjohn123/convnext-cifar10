# 06_report —— 报告目录（英文 Word / PDF 报告 + 生成脚本）

> **归属任务**：**T8（报告撰写）**，素材来自 T1–T7（要求解析 / 检索记录 / 代码 / 数据 / 结果）
> 作业主题：**AIassign2**（正文标题 **AI Assignment #1**）｜课程：**CISC3024 Pattern Recognition**
> 算法：**ConvNeXt**（Deep CNN，CVPR 2022，arXiv:2201.03545）｜任务：**CIFAR-10 图像分类**

---

## 1. 交付文件

| 文件 | 大小 | 说明 |
|---|---|---|
| `Assignment2_Report_CHAOWENJIE_UC325315.docx` | 544 KB | **正式交付件（Word）**：20 页、约 7,300 词、13 张表、4 张图，目录（TOC）域已由 Word 更新填充 |
| `Assignment2_Report_CHAOWENJIE_UC325315.pdf` | 851 KB | 同一内容的 **PDF 版本**（提交允许 Word 或 PDF，二者任选其一） |
| `report_builder/build_report.js` | 59 KB | 早期生成脚本（docx-js）：**已与当前 docx 不同步**——封面精简、去 AI 化改写、公式排版、附录删减等后续修改均直接在 docx 上完成，未回写该脚本 |

> 报告语言为**英文**（作业要求 **CON-5**），排版规格：US Letter、1 英寸页边距、Arial 正文 11 pt，
> 页眉写明作业与算法，页脚为 "Page X of Y"。已通过 OOXML 校验（`All validations PASSED`，595 段落）。

---

## 2. 报告结构与作业 6 项要求的对应关系

报告严格按 `01_requirements/report_requirements.txt` 的 6 项要求分章：

| 要求（原文要点） | 报告章节 | 主要素材来源 |
|---|---|---|
| (1) how you ask AI tools to find the algorithm | **§1**（含约束清单表、候选池表、来源核验表、提问方法） | `02_ai_chats/I0_algorithm_search_chat.md`、`algorithm_candidates.md` |
| (2) algorithm description | **§2**（现代化路线图、ConvNeXt Block 公式、预设表、32×32 适配、训练配方） | `02_ai_chats/convnext_algorithm_profile.md`、`03_code/src/models/convnext.py` |
| (3) how AI implements the algorithm | **§3**（协作协议、工程结构表、关键工程决策、验证清单、三个真实硬件问题及其修复、可复现措施） | `03_code/README.md`、05_results 的日志与检查点 |
| (4) experiment settings and results | **§4**（数据结构、设置表、主结果表、逐类结果与误差分析、训练行为、fp16 对照实验、CPU 基线、讨论与局限） | `05_results/full_convnext_tiny_cifar10/`、`05_results/mvp_atto_subset5k/`、`05_results/_failed_runs/` |
| (5) what you have learnt | **§5**（6 个小节：AI 与人工验证的分工、数值精度、平台差异、可复现性、提问方法、下次改进） | 全过程复盘、`02_ai_chats/ai_collaboration_plan.md` |
| (6) a webpage link of your source codes | **§6**（仓库链接 + 目录说明 + 运行命令） | `03_code/`；链接已发布：https://github.com/umjohn123/convnext-cifar10 |

另附 **References**（13 条：ConvNeXt 论文与官方实现、CIFAR-10、Mixup、CutMix、DropPath、AdamW、PyTorch 文档等）。

---

## 3. 报告中的表格与图（编号可直接引用）

| 编号 | 内容 | 编号 | 内容 |
|---|---|---|---|
| 表 1 | 从作业提取的可核验约束 | 表 8 | 三个真实硬件问题及修复 |
| 表 2 | 候选算法池与决策依据 | 表 9 | 主实验配置 |
| 表 3 | 算法来源核验 | 表 10 | 主实验结果（单视角 / TTA） |
| 表 4 | 现代化路线图的 8 项设计 | 表 11 | 逐类准确率 |
| 表 5 | 模型预设与参数量 | 表 12 | fp16 与 bf16 对照实验 |
| 表 6 | 代码工程结构 | 表 13 | CPU 基线（流水线验证） |
| 表 7 | 对生成代码执行的校验 | 图 1–4 | 训练曲线、混淆矩阵、逐类准确率、数据样例 |

> 图 1–4 直接取自 `05_results/full_convnext_tiny_cifar10/figures/`（工程自动生成，非手工绘制）；
> 表中的全部数值与 `05_results/.../metrics/` 中的结果文件一致（Top-1 92.66 %、TTA 92.87 %、Top-5 99.56 %）。

---

## 4. 提交前检查（已完成项与剩余步骤）

| # | 项目 | 位置 | 状态 |
|---|---|---|---|
| 1 | 填写**姓名与学号** | 报告第 1 页 "Author" 行 | ✅ 已完成：`<Chao Wen Jie > (<UC325315>)` |
| 2 | 替换**源码仓库链接** | 报告 §6 | ✅ 已完成：https://github.com/umjohn123/convnext-cifar10（PDF 中为可点击链接） |
| 3 | 更新目录 | 报告第 2 页 | ✅ 已完成：目录域已刷新，页码与正文一致 |

---

## 5. 重新生成报告

报告不是手工排版产物，而是由脚本生成 —— 任何数据更新后都可以一键重建：

```bash
cd 06_report/report_builder
npm install docx                     # 首次需要（约 10 MB）

# 1) 修改脚本中的内容（如作者姓名、仓库链接、指标数值），然后重建 .docx
node build_report.js                 # 输出到 ../Assignment2_Report_CHAOWENJIE_UC325315.docx

# 2) 校验 OOXML（需 pip install defusedxml lxml）
set PYTHONUTF8=1
python "%USERPROFILE%\.codebuddy\plugins\marketplaces\codebuddy-plugins-official\plugins\docx\scripts\office\validate.py" ../Assignment2_Report_CHAOWENJIE_UC325315.docx

# 3) 用本机 MS Word 更新目录并导出 PDF（也可直接在 Word 中"另存为 PDF"）
```

> 生成脚本中的指标数值来自 `05_results/full_convnext_tiny_cifar10/metrics/`；
> 若重跑实验得到新结果，只需同步这些数值再执行上面的第 1 步。

---

## 6. 与其它目录的引用关系

| 关联 | 说明 |
|---|---|
| ← 要求 | `01_requirements/report_requirements.txt`（6 项要求）、`requirement_summary.md`（约束 CON-1～CON-8） |
| ← 素材 | `02_ai_chats/`（§1、§2、§5）、`03_code/`（§3、§6）、`04_data/`（§4.1）、`05_results/`（§4 与图 1–4） |
| → 提交 | 报告定稿（Word/PDF，内嵌源码链接） |
