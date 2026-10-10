# screenshots —— AI 对话与运行过程截图证据

| 项目 | 内容 |
|---|---|
| 文档名称 | 截图证据目录说明 |
| **归属任务** | **T3 —— AI 检索算法对话与记录**（并服务于 T4、T6 的过程证据） |
| 归档位置 | `02_ai_chats/screenshots/` |
| 依据规范 | `02_ai_chats/ai_collaboration_plan.md` 第 2.3 节「证据留存规范」 |

> **说明**：本目录用于存放**图像证据**（对话界面截图、运行界面截图）。
> 文本形式的对话记录存放于上级目录 `02_ai_chats/*.md`；
> 截图需由操作者在交互界面中自行截取后放入本目录（AI 无法代为截屏）。

---

## 1. 命名规范

```
screenshots/
├── 01_<迭代编号>_<主题>.png
├── 02_...
└── ...
```

* 前缀两位数字用于控制展示顺序；
* 迭代编号与 `02_ai_chats/ai_collaboration_plan.md` 中的 I0–I4 一致；
* 主题使用简短的英文小写下划线形式（如 `algorithm_search`、`first_run`）。

---

## 2. 建议补充的截图清单

| 建议文件名 | 内容 | 对应报告项 |
|---|---|---|
| `01_I0_algorithm_search.png` | 向 AI 提问"检索 2022–2026 年的 Deep CNN / Deep Autoencoder + CV 应用"的对话界面 | 第 1 项 |
| `02_I0_algorithm_verification.png` | AI 给出 ConvNeXt 论文出处、年份、官方实现的对话界面 | 第 1、2 项 |
| `03_I0_code_generation.png` | 要求 AI 生成 ConvNeXt 实现代码的对话界面 | 第 3 项 |
| `04_I0_error_fixing.png` | 出现报错后向 AI 提供报错栈、获得修正代码的对话界面 | 第 3 项 |
| `05_I0_first_run.png` | `python train.py ...` 在终端 / Colab 中的运行输出画面 | 第 3、4 项 |
| `06_I0_training_curves.png` | 训练结束后生成的 `training_curves.png` 展示画面 | 第 4 项 |
| `07_I0_results_files.png` | `05_results/mvp_atto_subset5k/` 的目录结构画面 | 第 4 项 |
| `08_I4_colab_full_run.png` | 全量实验在 Colab GPU 上运行的画面 | 第 4 项 |
| `09_I4_source_link.png` | 源码托管仓库页面，可看到仓库 URL | 第 6 项 |

---

## 3. 记录要求

1. 截图应能**同时看到**提问内容与 AI 回答（体现"由 AI 完成"）；
2. 涉及报错修复的截图，应能看清**报错信息**与**修正后代码**；
3. 运行类截图建议保留**时间戳或终端提示符**，便于与 `05_results/*/logs/` 中的日志时间对齐；
4. 图像建议为 PNG，宽度不低于 1000 px，避免过度压缩导致文字不可读。

---

## 4. 对应的文本记录（已归档）

| 文本记录 | 路径 |
|---|---|
| I0 算法检索对话记录 | `../I0_algorithm_search_chat.md` |
| I0 迭代小结 | `../I0_summary.md` |
| 候选算法清单 | `../algorithm_candidates.md` |
| ConvNeXt 算法档案 | `../convnext_algorithm_profile.md` |
| AI 协作方案与迭代计划 | `../ai_collaboration_plan.md` |
