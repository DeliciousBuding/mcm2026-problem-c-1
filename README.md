# 📊 DWTS Fan Vote Inversion Framework

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![MCM/ICM](https://img.shields.io/badge/MCM%2FICM-2026%20Problem%20C-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

**MCM/ICM 2026 Problem C - Dancing with the Stars**

*Inferring Fan Votes from Elimination Data: A Dual-Core Inversion Framework with Mismatch Detection and Mechanism Design*

[📄 English Paper](paper/en/PaperC/) · [📄 中文论文](paper/zh/) · [📚 文档](docs/) · [🚀 快速开始](#-快速开始)

</div>

---

## 🎯 项目概述

本项目是 **MCM/ICM 2026 竞赛问题 C** 的完整解决方案，研究对象是美国知名真人秀节目 **Dancing with the Stars (DWTS)**。我们开发了一套完整的数学建模框架，用于从有限的淘汰数据中推断观众投票分布，并设计更公平的评分机制。

### 研究目标

| Task | 目标 | 核心方法 | 关键产出 |
|:----:|------|---------|---------|
| **Q1** | 推断观众投票分布 | LP/MILP 反演 + 贝叶斯采样 | 投票区间、后验分布、不确定性量化 |
| **Q2** | 反事实规则评估 | 规则引擎 + 蒙特卡洛模拟 | 淘汰矩阵、民主赤字、敏感性分析 |
| **Q3** | 特征归因分析 | XGBoost + SHAP + 前向链式验证 | 特征重要性、交互效应、生存因子 |
| **Q4** | 机制设计优化 | DAWS 动态权重 + 帕累托前沿 | 最优权重策略、鲁棒性验证 |

---

## 📁 目录结构

```
ProblemC-1/
├── README.md                          # 项目说明（本文件）
│
├── data/                              # 数据目录
│   ├── 2026_MCM_Problem_C_Data.csv    # 官方主数据集（33季完整数据）
│   └── 额外补充数据集/                 # 扩展数据（外部验证用）
│
├── src/                               # 源代码目录
│   ├── dwts_model/                    # 核心模型代码
│   │   ├── config.py                  # 全局配置参数
│   │   ├── paper_palette.py           # 论文统一配色系统
│   │   ├── etl/                       # 数据工程模块
│   │   ├── engines/                   # 反演引擎（LP/MILP）
│   │   ├── sampling/                  # 贝叶斯采样模块
│   │   └── analysis/                  # 分析模块
│   └── scripts/                       # 执行脚本
│       ├── run_full_pipeline.py       # ⭐ 完整流程一键执行
│       ├── generate_flowchart.py      # 流程图生成脚本
│       └── toy_rule_engine.py         # 规则引擎演示
│
├── paper/                             # 论文目录（中英文共享资源）
│   ├── main.tex                       # 英文论文主文件
│   ├── main_zh.tex                    # 中文论文主文件
│   ├── ref.bib                        # 参考文献
│   ├── sections/                      # 共享章节
│   ├── appendices/                    # 附录
│   └── figures/                       # 共享图表（唯一图表源）
│
├── docs/                              # 技术文档
│   ├── 01_算法设计详解.md             # LP/MILP 反演算法详解
│   ├── 02_综合总结.md                 # 模型设计思路与结果
│   ├── 03_运行与复现指南.md           # 复现步骤与环境配置
│   ├── 04_优化分析.md                 # 性能优化与参数调优
│   ├── 05_外部验证.md                 # 外部数据验证分析
│   ├── 06_图表规范.md                 # 论文图表视觉系统
│   └── 07_开发规范.md                 # Git 工作流与提交规范
│
├── outputs/                           # 分析结果输出
│   ├── *.csv                          # 数据表格
│   ├── *.tex                          # LaTeX 表格
│   └── FULL_RUN_SUMMARY.md            # 运行摘要报告
│
└── archive/                           # 归档文件（历史版本、废弃文件）
```

> **设计说明**：采用"单图表源 + 多主文件"架构，`paper/figures/` 是唯一图表输出目录，中英文论文共享引用，避免重复同步。

---

## 🧠 核心模型架构

### 模型总览

```
                           ┌─────────────────────────────────────────┐
                           │       DWTS Fan Vote Inversion Framework │
                           └─────────────────────────────────────────┘
                                              │
        ┌─────────────────────────────────────┼─────────────────────────────────────┐
        │                                     │                                     │
        ▼                                     ▼                                     ▼
┌───────────────────┐              ┌───────────────────┐              ┌───────────────────┐
│   Q1: Inversion   │              │  Q2: Counterfact  │              │  Q3-Q4: Design    │
│   LP/MILP Engine  │──────────────│   Rule Engine     │──────────────│   DAWS System     │
│   + Bayesian      │              │   + MC Simulate   │              │   + Pareto        │
└───────────────────┘              └───────────────────┘              └───────────────────┘
```

### 双核反演引擎（Dual-Core Inversion Engine）

根据 DWTS 历史规则变化，我们设计了自适应的双核引擎：

| 赛季类型 | 规则描述 | 反演方法 | 数学形式 |
|:-------:|---------|:-------:|---------|
| **百分制** (S3–S27) | 评委分×α + 粉丝票×(1-α) | **线性规划 (LP)** | $\min \|f\|$ s.t. $Af \leq b$ |
| **排名制** (S1–S2, S28+) | 评委排名 + 粉丝排名 → 淘汰 | **混合整数规划 (MILP)** | $\min c^Tx$ s.t. $Ax \leq b, x \in \mathbb{Z}$ |

### 贝叶斯采样与后验推断

```
硬约束反演 ──→ 可行区间 [L_i, U_i]
                    │
                    ▼
    截断 Dirichlet 先验 + Hit-and-Run 采样
                    │
                    ▼
    时间平滑先验（Gaussian Random Walk）
                    │
                    ▼
         后验均值 + 95% HDI 区间
```

### 规则引擎（Rule Engine）

支持三种投票规则的模拟与反事实分析：

| 规则 | 代码标识 | 描述 |
|-----|---------|------|
| **百分制规则** | `percent_rule` | α·评委% + (1-α)·粉丝%，最低分淘汰 |
| **排名制规则** | `rank_rule` | 排名相加，Bottom-2 中淘汰一人 |
| **评委拯救** | `rank_with_judge_save` | 排名制 + 评委可拯救（J0/J1/J2 三种模式） |

**评委拯救模式**:
- `J0`: 随机拯救（基线）
- `J1`: Softmax 概率拯救（按评委分数软化）
- `J2`: 确定性拯救（始终拯救高分选手）

---

## 📄 论文结构

**标题**: *Inferring Fan Votes from Elimination Data: A Dual-Core Inversion Framework with Mismatch Detection and Mechanism Design for Dancing with the Stars*

### 章节概览

| 章节 | 文件 | 内容摘要 |
|:---:|------|---------|
| **Abstract** | `00_abstract.tex` | 摘要与关键词 |
| **Memo** | `memo.tex` | 给 DWTS 制作人的执行备忘录 |
| **§1** | `01_intro.tex` | 问题背景、研究动机、贡献总结 |
| **§2** | `02_assumptions.tex` | 核心假设与合理性论证 |
| **§3** | `03_notations.tex` | 数学符号定义表 |
| **§4** | `04_model1.tex` | **粉丝票反演**（LP/MILP + 贝叶斯） |
| **§5** | `05_model2.tex` | **存活分析 + 特征归因**（XGBoost/SHAP） |
| **§6** | `06_model3.tex` | **机制设计**（DAWS 动态权重系统） |
| **§7** | `07_sensitivity.tex` | 敏感性分析与鲁棒性检验 |
| **§8** | `08_evaluation.tex` | 模型评估与验证 |
| **§9** | `09_conclusion.tex` | 结论、建议与未来工作 |

---

## 📊 核心结果摘要

基于 8000 样本完整运行的关键指标：

### Q1 - 反演与不确定性量化

| 指标 | 数值 | 说明 |
|-----|:----:|------|
| 平均接受率 | 38.3% | 可行解空间质量 |
| 区间宽度均值 | 0.849 | 后验不确定性 |
| PPC Top-3 覆盖率 | 96.6% | 后验预测检验通过率 |
| PPC Brier Score | 0.019 | 概率校准质量 |

### Q2 - 反事实规则评估

| 规则 | 技能对齐 | 观众代理 | 稳定性 |
|-----|:-------:|:-------:|:-----:|
| 百分制 | 0.462 | **0.662** | **0.726** |
| 排名制 | 0.453 | 0.481 | 0.577 |
| 评委拯救 | 0.453 | 0.307 | 0.540 |

### Q3-Q4 - ML/XAI 与机制设计

| 指标 | 数值 |
|-----|:----:|
| 前向链式验证 AUC | 0.984 |
| 前向链式验证 Brier | 0.018 |
| DAWS 最优 α | 0.8 |
| 噪声翻转率 | 1.88% |

---

## 📁 关键输出文件

### 数据表格（`outputs/`）

| 文件 | 说明 | 对应任务 |
|-----|------|:-------:|
| `fan_vote_intervals.csv` | LP/MILP 求解的粉丝票可行区间 | Q1 |
| `fan_vote_posterior_summary.csv` | 后验均值与 95% HDI 区间 | Q1 |
| `ppc_summary.csv` | 后验预测检验统计 | Q1 |
| `counterfactual_matrix.csv` | 反事实淘汰评估矩阵 | Q2 |
| `democratic_deficit.csv` | 民主赤字（KL 散度）| Q2 |
| `save_sensitivity.csv` | 评委拯救敏感性分析 | Q2 |
| `forward_chaining_metrics.csv` | 前向链式验证指标 | Q3 |
| `shap_importance.csv` | SHAP 特征重要性排序 | Q3 |
| `pareto_frontier.csv` | 帕累托前沿解集 | Q4 |
| `daws_weekly.csv` | DAWS 周度权重输出 | Q4 |

### 可视化图表（`figures/`）

| 文件 | 说明 | 论文引用 |
|-----|------|---------|
| `fig_q1_hdi_band.pdf` | HDI 波段图（投票不确定性）| Fig.4 |
| `fig_q1_uncertainty_heatmap.pdf` | 不确定性热力图 | Fig.6 |
| `fig_q1_ppc_metrics.pdf` | 后验预测检验图 | Fig.7 |
| `fig_q2_counterfactual_matrix.pdf` | 反事实淘汰热力图 | Fig.8 |
| `fig_q2_democratic_deficit.pdf` | 民主赤字可视化 | Fig.10 |
| `fig_q3_forward_chaining.pdf` | 前向链式验证曲线 | Fig.9 |
| `fig_shap_summary.pdf` | SHAP 特征重要性 | Fig.10-11 |
| `fig_q4_pareto_frontier.pdf` | 帕累托前沿图 | Fig.13 |
| `fig_ternary_tradeoff.pdf` | 三元权衡可视化 | Fig.13 |
| `fig_dwts_flowchart_vector.pdf` | 模型架构流程图 | Fig.1 |
| `fig_sankey_audit.pdf` | 规则审计桑基图 | Fig.2 |

---

## 🚀 快速开始

### 环境要求

| 依赖 | 版本要求 | 用途 |
|-----|---------|------|
| Python | ≥ 3.10 | 运行环境 |
| numpy, pandas, scipy | 最新稳定版 | 数值计算 |
| matplotlib, seaborn | 最新稳定版 | 可视化 |
| xgboost, shap | 最新稳定版 | 机器学习 / 特征分析 |
| LaTeX (xelatex + biber) | — | 论文编译（可选） |

### 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install numpy pandas scipy matplotlib seaborn xgboost shap
```

### 一键执行

```bash
# 完整流程（推荐）
python src/scripts/run_full_pipeline.py --samples 2000 --smooth-sigma 0.15

# 快速测试（减少采样）
python src/scripts/run_full_pipeline.py --samples 500 --skip-compile

# 仅生成图表（跳过 ML 和论文编译）
python src/scripts/run_full_pipeline.py --skip-ml --skip-compile
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--samples` | 2000 | 每周采样数量 |
| `--smooth-sigma` | 0.15 | 时间平滑强度（越小越平滑） |
| `--skip-ml` | False | 跳过 XGBoost + SHAP 分析 |
| `--skip-compile` | False | 跳过 LaTeX 论文编译 |

---

## ✨ 技术亮点

| 特性 | 说明 |
|:----:|------|
| 🔧 **双核反演引擎** | 百分制用 LP，排名制用 MILP，统一接口自动适配 |
| 📊 **贝叶斯不确定性量化** | Hit-and-Run 采样 + Gaussian RW 先验，提供置信区间 |
| ✅ **规则一致性过滤** | 确保采样能复现实测淘汰结果 |
| 📐 **信息论度量** | 用 KL 散度量化"民主赤字"（评委 vs 观众偏好差异）|
| ⚖️ **帕累托机制设计** | DAWS 系统在公平性与参与度间寻找最优平衡 |
| 🔄 **前向链式验证** | 时序分割避免数据泄露，AUC 达 0.984 |
| 🎨 **工业级可视化** | 统一配色系统，符合学术论文规范 |

---

## 📚 文档索引

| 文档 | 说明 |
|-----|------|
| [01_算法设计详解](docs/01_算法设计详解.md) | LP/MILP 反演算法详解 |
| [02_综合总结](docs/02_综合总结.md) | 模型设计思路与结果总结 |
| [03_运行与复现指南](docs/03_运行与复现指南.md) | 详细的复现步骤与环境配置 |
| [04_优化分析](docs/04_优化分析.md) | 性能优化与参数调优策略 |
| [05_外部验证](docs/05_外部验证.md) | 使用外部数据的验证分析 |
| [06_图表规范](docs/06_图表规范.md) | 论文图表视觉系统设计规范 |
| [07_开发规范](docs/07_开发规范.md) | **Git 工作流、分支策略、提交规范** |

---

## ❓ 常见问题

<details>
<summary><b>Q: 运行时间过长怎么办？</b></summary>

减少采样数量或跳过 ML 分析：
```bash
python src/scripts/run_full_pipeline.py --samples 500 --skip-ml
```
</details>

<details>
<summary><b>Q: 后验分布过于分散（平滑不足）？</b></summary>

增大采样数或降低 σ 值（更强平滑）：
```bash
python src/scripts/run_full_pipeline.py --samples 4000 --smooth-sigma 0.10
```
</details>

<details>
<summary><b>Q: SHAP 图生成失败？</b></summary>

确认 xgboost 和 shap 版本兼容：
```bash
pip install xgboost==2.0.0 shap==0.44.0
```
</details>

<details>
<summary><b>Q: LaTeX 编译失败？</b></summary>

检查 LaTeX 工具链是否正确安装：
```bash
xelatex --version
biber --version
```
或使用 `--skip-compile` 参数跳过编译。
</details>

<details>
<summary><b>Q: MILP 求解器警告？</b></summary>

对于大规模周（选手数 > 10），MILP 引擎可能回退到启发式方法，这是预期行为，不影响结果质量。
</details>

---

## 📜 许可证

本项目仅用于 MCM/ICM 2026 竞赛学术用途。

---

<div align="center">

**Team #2617892** | MCM/ICM 2026 Problem C

*Built with ❤️ for Mathematical Modeling*

</div>
