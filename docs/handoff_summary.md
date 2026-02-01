# Handoff Summary (Codex)

Date: 2026-01-31
Project: d:\MCMICM\MCMICM\develop-problem-C

## User Intent & Constraints (current thread)
- Role: paper visualization director/engineer; only edit figures/scripts/LaTeX for visual system, no model/data changes.
- Recent scope: **ONLY Fig.1 (DWTS Competition Process)** edits in `scripts/generate_flowchart.py` / `generate_show_flowchart()`.
- Hard rules (active):
  - Do not modify other figures or scripts.
  - Do not alter model logic or data.
  - Keep color system/style consistent.
  - Fix semantics: “performance → judges/fans → combined score”, no dangling arrows.
  - No new nodes or explanatory text.
  - Bottom-2 label should not read as a question.
  - Audit fusion text must not be overlapped by arrows.
  - Overall readability improved by **+1 pt** (latest requirement); earlier bumps were higher, now normalized to +1.
- Deliverables for Fig.1: `figures/fig_dwts_show_process.pdf` + 2x PNG.
origin contents:
你是“现代工业极简风（Apple/Tesla/Google Research）技术白皮书”的首席可视化工程师 + 排版设计师。你的任务：在不改任何模型、数据、结论的前提下，对本项目论文的所有图与排版进行“统一视觉系统重构”，输出新的高质感 PDF（目标：更像工业级技术白皮书、更像 O 奖审美），并通过“视觉验收”确认问题被修复。

【重要约束】
1) 你必须用“视觉方式”阅读现有 main.pdf：尝试把 PDF 渲染成图片逐页查看（例如用 PyMuPDF/fitz 或 pdf2image），并基于图形视觉观感指出问题；禁止用 OCR/图片转文字去理解图的内容。 
2) 不允许修改任何算法结论、统计结果、数据点；只能改：配色、字体、布局、线条、图例、标注、留白、刻度、图类型外观（例如热图离散化、强调策略等）。
3) 所有改动必须在代码层面可复现：修改绘图脚本/样式文件/LaTeX 排版参数，确保一键生成一致风格的所有图。

【工作流（必须按顺序执行，不得跳过）】
Step 0 — 建立基线与资产清单
0.1 渲染现有 main.pdf 每一页为 PNG（至少 2x 分辨率）存到 outputs/audit_before/，并建立一个“视觉问题清单”（按 Figure 编号）。
0.2 扫描仓库：定位所有生成 figures 的脚本（例如 scripts/、src/、final_analysis.py、visualization*.py 等），建立 Figure 编号 -> 生成脚本/函数 -> 输出文件名 的映射表。
0.3 找到 LaTeX 中 figures 的引用与排版参数（图宽、caption、浮动体间距等），记录当前设置。

Step 1 — 基于视觉审阅 main.pdf，逐图指出“具体问题”（必须覆盖）
你必须至少识别并修复下列“已知视觉痛点”（这些痛点来自对现有 main.pdf 的真实视觉观感）：
A) Fig.1 / Fig.3 流程图/架构图：字体偏小、箭头太细、信息密度不均、整体不够“工业极简”；需要更强的栅格对齐与层级（标题/模块/箭头/注释）。
B) Fig.2（Audit Sankey/Flow）：目前更像“学术默认图”，存在纹理/填充（hatch）显得花；图例与标注不够克制；整体缺少现代白皮书的轻量感与对齐感。
C) Fig.4（Uncertainty Heatmap）：色阶连续但“焦点不明确”，刻度与轴标签偏拥挤，小字多；需要更“结构化”的表达（强调 band / 区域模式），并降低视觉噪声。
D) Fig.9（Forward-Chaining Validation）：存在“靠近 1.0 的拥挤/截断风险”（需要检查是否有点/线超出边界或贴边），图例占位不优雅；线型/标记偏多，缺少主次。
E) Fig.10/11（SHAP）：过度“机器学习风”，特征过多导致拥挤；颜色高饱和、对比太强；需要降噪（Top-K 强调，其余灰）、字体可读性提升、色条改为低饱和单色系。
F) Fig.13（Pareto + Ternary）：当前标签/点/图例容易重叠，信息密度高但不“可读”；需要重新布局、减少标注、用“推荐点”突出结论；并修复文中出现 “Fig. ?? shows ...” 之类引用错误（如存在）。
G) 统一问题：全论文图的字体族、字号、线宽、网格、颜色系统不统一，缺少“工业设计一致性”。

Step 2 — 设计并落地“统一视觉系统”（强制规范）
你必须创建一个可复用的 Matplotlib 样式模块（例如 figures/style_modern.py 或 mplstyle 文件），并让所有绘图脚本统一引用。风格目标：极简、克制、科技工业感；主色 + 灰阶；少用装饰。

2.1 颜色系统（强制：1 主色 + 灰阶；禁止彩虹）
- 主色 Primary：#1F2933（深蓝灰）
- 强调 Accent（只用于“推荐点/关键线/关键区域”）：#2563EB（低饱和蓝）
- 灰阶：#111827 #6B7280 #D1D5DB #F9FAFB
规则：同一张图最多允许“主色 + 1 个强调色 + 灰阶”，其余全部灰；禁止高饱和红黄绿。

2.2 字体与线条（强制）
- 字体：优先 Inter / Helvetica / Arial（sans-serif），不使用花哨衬线；全论文一致。
- 默认字号：轴标签 10，刻度 9，图例 9，标题 11（必要时标题 12），caption 不改（LaTeX 里控制）。
- 线宽：主线 1.8，次线 1.2，网格线 0.6；坐标轴 spine 0.8。
- 网格：仅 y 方向细灰网格（alpha<=0.15），避免满屏网格。

2.3 图例与标注（强制）
- 图例必须“简短、贴边、不遮挡数据”；优先放外侧或右上角空白区；尽量一行。
- 标注只保留 1~3 个关键点（例如推荐点/极值点），其余不要标注。

2.4 留白与版式（强制）
- 画布留白：tight_layout / constrained_layout，但要确保标题、轴标签、图例不拥挤。
- 输出分辨率：dpi >= 300；字体矢量化（pdf/svg 优先）。
- 所有图保存时边距一致（pad_inches 合理）。

Step 3 — 按图类型执行“改造策略”（必须做）
3.1 流程图/架构图（Fig.1/Fig.3）
- 重新设定字号（至少比现在大一档），箭头线宽加粗（>=1.6），统一圆角矩形风格、统一间距；
- 使用主色描边 + 浅灰填充（或白底描边），禁止花纹；
- 栅格对齐：模块左对齐/居中对齐一致；箭头尽量水平/垂直，少斜线；
- 输出为矢量（PDF/SVG）嵌入 LaTeX。

3.2 流/桑基/条形结构图（Fig.2）
- 去除 hatch/纹理，改为纯色块（浅灰/主色），用“线条与留白”表达层级；
- 增大关键数字字号，减少多余标注；
- 图例改为小型、灰阶；确保不挡任何数据。

3.3 热图（Fig.4 / Fig.6 类矩阵）
- 色盘改为“灰 -> 主色”的单色系；可把连续色离散为 5 档（视觉更像结构带）；
- 轴刻度简化（例如只标注关键季/关键周，或每隔 N 个显示），避免小字堆叠；
- 强化“band/区域模式”：可加浅灰分隔线或淡淡框线强调区块；
- colorbar 刻度减少，并用简洁标签（Low/High 或 0~1 少量刻度）。

3.4 折线/验证曲线（Fig.9 / Fig.7 / Fig.12）
- 主线（关键指标）用主色，次线用中灰；marker 极少或只用于关键点；
- 确保 y 轴范围留出上边距（避免 1.0 贴边/截断），必要时设置 ylim 上界 > 1.0 的一点点（例如 1.02）；
- 图例放到不遮挡处，或置于外侧；去掉不必要的边框。

3.5 SHAP（Fig.10/11）
- 只保留 Top 10（最多 Top 12）特征显示，其余合并为 “Others” 或直接不画；
- Top 3 用主色，其余用浅灰；色条从灰到主色，避免红蓝强对比；
- 调整图尺寸与边距，让特征名不挤；必要时横向更宽；
- 字体增大一档；确保打印可读。

3.6 Pareto + Ternary（Fig.13）
- 彻底解决“标签/点重叠”：减少标注数量，只标推荐点与少数代表点；
- 可把左侧 Pareto 与右侧 Ternary 拆为两个子图但保持统一风格，或维持同页但扩大画布；
- 推荐点用强调色 + 清晰注释；其余点灰色小点；
- 若存在正文 “Fig. ?? ” 引用错误，必须定位 LaTeX 引用并修复（label/ref）。

Step 4 — LaTeX 排版同步升级（必须做）
4.1 统一 figure 宽度策略（例如大图 \linewidth，小图 0.85\linewidth），避免忽大忽小。
4.2 增加图上下间距（\textfloatsep / \floatsep / \intextsep 适度调大），让页面更“呼吸”。
4.3 Caption 规则：caption 第一行必须是“结论句”（不是 Figure X shows...）。保持英文简洁有力。
4.4 检查所有图编号引用（\label/\ref）无 “??”。

Step 5 — 重新生成、编译与“视觉验收”（必须输出证据）
5.1 运行完整 pipeline 重新生成所有 figures（确保全部采用统一 style_modern）。
5.2 重新编译 LaTeX 生成 new_main.pdf。
5.3 将 new_main.pdf 同样逐页渲染到 outputs/audit_after/（2x 分辨率），并对比 before/after（至少对比含 Fig.1/2/4/9/10/13/14 的页面）。
5.4 输出一份简短的 CHANGELOG（markdown）：逐条列出每个 Figure 做了什么、解决了哪个视觉问题。

【最终交付物】
- 新的 PDF：new_main.pdf（或覆盖 main.pdf，但需保留备份）
- 统一样式文件：style_modern.py 或 .mplstyle
- 更新后的绘图脚本与 LaTeX 源文件
- before/after 渲染图片目录 + CHANGELOG

【硬性验收标准（不满足就继续迭代）】
- 全论文图的字体、线宽、颜色系统完全一致；
- 所有图在 100% 缩放下可读：刻度、图例、关键标注不拥挤；
- 不存在任何元素遮挡数据（图例/注释不压点、不挡条形）；
- 不存在贴边截断（尤其 y=1.0 附近的曲线/点）；
- Fig.13 不再出现标签重叠；正文无 “Fig. ??”；
- 视觉观感：整体更“克制、干净、工业化”，更像技术白皮书而非默认学术图。

现在开始执行：先完成 Step 0 的 PDF 逐页渲染与“逐图视觉问题清单”，再进入后续步骤。不要跳过。


## Step 0 baseline artifacts
- Audit render of original PDF pages (before): `outputs/audit_before/page_01.png` … `page_23.png`.
- Visual issue list saved: `outputs/audit_before/visual_issues_by_figure.md`.

## Figure script mapping (LaTeX → generator)
- Fig.1 `fig_dwts_show_process.pdf` → `scripts/generate_flowchart.py` → `generate_show_flowchart()`
- Fig.2 `fig_sankey_audit.pdf` → `scripts/run_full_pipeline.py` → `plot_sankey_audit()`
- Fig.3 `fig_dwts_flowchart_vector.pdf` → `scripts/generate_flowchart.py` → `generate_analytical_framework()`
- Fig.4 `fig_q1_uncertainty_heatmap.pdf` → `scripts/run_full_pipeline.py` → `plot_uncertainty_heatmap()`
- Fig.5 `fig_q1_hdi_band.pdf` → `scripts/run_full_pipeline.py` → `plot_hdi_band()`
- Fig.6 `fig_q2_counterfactual_matrix.pdf` → `scripts/run_full_pipeline.py` → `plot_counterfactual_matrix()`
- Fig.7 `fig_q2_save_sensitivity.pdf` → `scripts/run_full_pipeline.py` → `plot_save_sensitivity()`
- Fig.8 `fig_q2_democratic_deficit.pdf` → `scripts/run_full_pipeline.py` → `plot_democratic_deficit()`
- Fig.9 `fig_q3_forward_chaining.pdf` → `src/dwts_model/analysis/ml_xai.py` → `_plot_forward_chaining()`
- Fig.10 `fig_shap_summary.pdf` → `src/dwts_model/analysis/ml_xai.py` → `_plot_shap_summary()`
- Fig.11 `fig_shap_interaction.pdf` → `src/dwts_model/analysis/ml_xai.py` → `_plot_shap_interaction()`
- Fig.12 `fig_q4_alpha_schedule.pdf` → `src/dwts_model/analysis/daws.py` → `plot_alpha_schedule()`
- Fig.13 `fig_q4_pareto_frontier.pdf` + `fig_ternary_tradeoff.pdf` → `scripts/run_full_pipeline.py` → `run_pareto_frontier()` / `plot_ternary_tradeoff()`
- Fig.14 `fig_q4_noise_robustness.pdf` → `src/dwts_model/analysis/daws.py` → `plot_noise_robustness()`
- Fig.15 `fig_q1_ppc_metrics.pdf` → `scripts/run_full_pipeline.py` → `plot_ppc_metrics()`

## LaTeX figure references/params (current)
- `paper/en/PaperC/sections/01_intro.tex`
  - Fig.1 width 0.95\textwidth; vspace -1.0em and -0.8em
  - Fig.2 width 0.95\textwidth; vspace -0.8em twice
- `paper/en/PaperC/sections/04_model1.tex`
  - Fig.3 width 0.9\textwidth
  - Fig.4 width 0.86\textwidth
  - Fig.5 width 0.86\textwidth
- `paper/en/PaperC/sections/05_model2.tex`
  - Fig.6 width 0.72\textwidth
  - Fig.7 width 0.68\textwidth
  - Fig.8 width 0.72\textwidth
- `paper/en/PaperC/sections/06_model3.tex`
  - Fig.9 width 0.78\textwidth
  - Fig.10 width 0.85\textwidth
  - Fig.11 width 0.7\textwidth
  - Fig.12 width 0.7\textwidth
  - Fig.13 two minipage each 0.48\textwidth
  - Fig.14 width 0.75\textwidth
- `paper/en/PaperC/sections/08_evaluation.tex`
  - Fig.15 width 0.78\textwidth
- `appendices/appendix_figures.tex`
  - `fig_shap_summary_full.pdf` width 0.85\textwidth (appendix not included in main)

## Visual issues list (by figure)
Saved in `outputs/audit_before/visual_issues_by_figure.md`. Key points include:
- Fig.1/3: hierarchy too flat, arrow thin, density and alignment issues.
- Fig.2: hatch texture noise; legend/labels not restrained.
- Fig.4: heatmap lacks focus; ticks crowded.
- Fig.9: near-1.0 crowding risk; legend placement.
- Fig.10/11: SHAP clutter, high saturation, too many features.
- Fig.13: label overlap risk; too dense.
- Global: inconsistent fonts/line widths/color usage.

## Current Fig.1 state (latest edits)
File: `scripts/generate_flowchart.py`
Function: `generate_show_flowchart()`
- **Layout changes** (within allowed scope for Fig.1 only):
  - Weekly Performance moved to top center (x=5.5, y=5.3).
  - Judges/Fans moved above Combined Score (x=4.3 and 6.7, y=4.6).
  - Combined Score moved down to y=3.8.
  - Scoring Era and Percent/Rank blocks shifted down to y=3.5/2.45.
  - Bottom 2 diamond moved to y=1.6 and text set to “Bottom\n2” (no question mark).
  - Outcome boxes moved down to y=0.7.
- **Arrow semantics** now: Weekly Performance → Judges/Fans; Judges/Fans → Combined Score; Combined Score → Scoring Era; Scoring Era → Percent/Rank → Bottom 2 → outcomes. No crossing arrows.
- **Audit fusion text** moved to (x=6.6, y=3.45) to avoid vertical arrow.
- **Font bump** set to `font_bump = 1` (overall +1 pt from base).
- Combined Score formula uses mathtext; text bbox pad for clipping: `bbox: pad=0.45`, `clip_on=False`.

## Files modified
- `scripts/generate_flowchart.py` (significant edits in `generate_show_flowchart()` + `draw_box()` updated to accept `text_kwargs`)
- Outputs regenerated:
  - `figures/fig_dwts_show_process.pdf`
  - `figures/fig_dwts_show_process.png` (2x)
  - Check image: `outputs/audit_before/fig1_check_100pct.png`
- Visual issues list saved:
  - `outputs/audit_before/visual_issues_by_figure.md`

## Notes / Known pitfalls
- Prior edits temporarily set font_bump=2; current set to 1 per latest request.
- Ensure no further edits outside Fig.1 until user confirms.
- Use PyMuPDF for PNG 2x generation (already used).

## User’s last explicit request (active)
- Provide summary for next window (this file) and stop.

