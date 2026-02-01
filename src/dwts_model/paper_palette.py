"""
DWTS 论文统一配色方案 (v1.0)

设计原则：
1. 语义先于美观：每种颜色只表达一种固定含义
2. 打印友好：灰度打印仍能区分（用线型/标记补偿）
3. 色盲友好：使用经过验证的 colorblind-safe 色板
4. 全篇一致：同类图使用同一模板

语义色彩绑定：
- 蓝 (#0072B2)：真实数据/主结果/默认模型输出
- 橙 (#E69F00)：推荐机制（DAWS）/最终建议点/最优平衡区
- 灰 (#7A7A7A)：基线（如 α=0.6 静态方案）/参考线
- 红 (#D55E00)：警示信息（淘汰/mismatch/异常点/违反约束）
"""

from typing import Optional

# ============================================================
# 核心语义色板 (v1.0 - 色盲友好)
# ============================================================

PALETTE = {
    # 主色系 - 语义绑定
    "primary": "#0072B2",        # 蓝 - 真实/主结果/默认模型输出
    "primary_dark": "#0B3C5D",   # 深蓝 - 强调主线/关键点
    "emphasis": "#E69F00",       # 橙 - 推荐机制/最终建议/最优区
    "warning": "#D55E00",        # 朱红 - 风险/淘汰/异常/违反约束
    "neutral": "#7A7A7A",        # 中性灰 - 基线/对照/辅助线
    "light_gray": "#D9D9D9",     # 浅灰 - 网格/置信带填充/背景区

    # 兼容旧代码的别名 (v0.x → v1.0)
    "proposed": "#0072B2",       # → primary (主机制)
    "baseline": "#7A7A7A",       # → neutral (基线用灰色)
    "fill": "#D9D9D9",           # → light_gray (填充)
    "aux": "#0B3C5D",            # → primary_dark (辅助深蓝)
    "accent": "#E69F00",         # → emphasis (强调)
    "warning2": "#D9D9D9",       # → light_gray (次级灰)
    
    # 旧蓝灰系别名（完全兼容旧代码）
    "deep_blue": "#0072B2",      # → primary
    "navy": "#0B3C5D",           # → primary_dark
    "cyan_blue": "#0072B2",      # → primary
    "light_blue": "#D9D9D9",     # → light_gray
    "pale_blue": "#E7EFF3",      # 极浅蓝
    "ref_gray": "#7A7A7A",       # → neutral
    "text_gray": "#555555",      # 文字灰
    "daws_gold": "#E69F00",      # → emphasis

    # 渐变用色（顺序色带）
    "seq_low": "#E7EFF3",        # 极浅蓝 - 热力图低值
    "seq_mid": "#9CB8C8",        # 浅蓝灰 - 热力图中值
    "seq_high": "#0072B2",       # 主蓝 - 热力图高值
}

# ============================================================
# 统一绘图标准 (v1.0)
# ============================================================

FIGURE_STANDARDS = {
    # 尺寸标准（英寸）- 宽高比约 1.68:1
    "figsize_standard": (6.4, 3.8),
    "figsize_compact": (6.4, 3.2),
    "figsize_wide": (8.0, 4.0),
    "figsize_square": (5.5, 5.5),

    # 字体标准（pt）- 与正文 10-11pt 匹配
    "title_fontsize": 10,          # 图内标题（建议少用）
    "label_fontsize": 9.5,         # 坐标轴标题
    "tick_fontsize": 8.8,          # 刻度文字
    "legend_fontsize": 8.8,        # 图例
    "annotation_fontsize": 8.5,    # 注释文字

    # 线宽标准（pt）
    "linewidth_main": 1.7,         # 主线宽
    "linewidth_secondary": 1.3,    # 次线宽
    "linewidth_baseline": 1.1,     # 基线/参考线
    "linewidth_grid": 0.5,         # 网格线
    "linewidth_border": 0.9,       # 轴线宽
    "linewidth_arrow": 0.9,        # 箭头线宽

    # 标记点标准（pt）
    "marker_size": 4.5,
    "marker_size_small": 3.5,
    "marker_size_large": 6.0,

    # 透明度标准
    "alpha_fill": 0.20,            # 置信区间/HDI填充
    "alpha_grid": 0.30,            # 网格透明度
    "alpha_scatter": 0.75,         # 散点透明度
    "alpha_zone": 0.12,            # 强调区域透明度

    # 刻度设置
    "tick_direction": "out",
    "tick_major_size": 3.5,
}


def get_paper_rcparams() -> dict:
    """
    返回论文级 matplotlib rcParams 配置 (v1.0)
    
    用法：
        import matplotlib as mpl
        from dwts_model.paper_palette import get_paper_rcparams
        mpl.rcParams.update(get_paper_rcparams())
    """
    return {
        # Figure
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.figsize": FIGURE_STANDARDS["figsize_standard"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # Fonts (serif for academic papers)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        # Font Sizes
        "axes.labelsize": FIGURE_STANDARDS["label_fontsize"],
        "axes.titlesize": FIGURE_STANDARDS["title_fontsize"],
        "xtick.labelsize": FIGURE_STANDARDS["tick_fontsize"],
        "ytick.labelsize": FIGURE_STANDARDS["tick_fontsize"],
        "legend.fontsize": FIGURE_STANDARDS["legend_fontsize"],

        # Lines
        "lines.linewidth": FIGURE_STANDARDS["linewidth_main"],
        "lines.markersize": FIGURE_STANDARDS["marker_size"],

        # Axes & Spines
        "axes.linewidth": FIGURE_STANDARDS["linewidth_border"],
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Grid
        "axes.grid": True,
        "grid.linewidth": FIGURE_STANDARDS["linewidth_grid"],
        "grid.alpha": FIGURE_STANDARDS["alpha_grid"],

        # Legend
        "legend.frameon": False,

        # Ticks
        "xtick.direction": FIGURE_STANDARDS["tick_direction"],
        "ytick.direction": FIGURE_STANDARDS["tick_direction"],
        "xtick.major.size": FIGURE_STANDARDS["tick_major_size"],
        "ytick.major.size": FIGURE_STANDARDS["tick_major_size"],
    }


# Backward compatibility alias
get_standard_rcparams = get_paper_rcparams


# ============================================================
# 语义化快捷访问
# ============================================================

VOTING_METHODS = {
    "percent": PALETTE["primary"],      # 百分比制 - 主蓝
    "rank": PALETTE["neutral"],         # 排名制 - 灰色基线
}

MECHANISMS = {
    "current": PALETTE["neutral"],      # 当前机制 - 灰
    "proposed": PALETTE["primary"],     # 新机制 - 蓝
    "daws": PALETTE["emphasis"],        # DAWS推荐 - 橙
    "soft_floor": PALETTE["primary_dark"],
}

DATA_STATES = {
    "match": PALETTE["primary"],        # 匹配 - 蓝
    "mismatch": PALETTE["warning"],     # 不匹配 - 红
    "uncertain": PALETTE["light_gray"], # 不确定 - 浅灰
    "recommended": PALETTE["emphasis"], # 推荐 - 橙
}

LEGEND_LABELS = {
    "percent": "Percent Method (S3-27)",
    "rank": "Rank Method (S1-2, S28+)",
    "mismatch": "Mismatch Detected",
    "proposed": "Proposed (DAWS)",
    "current": "Current/Baseline",
    "recommended": "Recommended",
}

# ============================================================
# 绘图样式预设
# ============================================================

LINE_STYLES = {
    "primary": {
        "color": PALETTE["primary"],
        "linewidth": FIGURE_STANDARDS["linewidth_main"],
        "linestyle": "-",
    },
    "baseline": {
        "color": PALETTE["neutral"],
        "linewidth": FIGURE_STANDARDS["linewidth_baseline"],
        "linestyle": "--",
    },
    "emphasis": {
        "color": PALETTE["emphasis"],
        "linewidth": FIGURE_STANDARDS["linewidth_main"],
        "linestyle": "-",
    },
    "warning": {
        "color": PALETTE["warning"],
        "linewidth": FIGURE_STANDARDS["linewidth_secondary"],
        "linestyle": "-",
    },
    "reference": {
        "color": PALETTE["neutral"],
        "linewidth": FIGURE_STANDARDS["linewidth_baseline"],
        "linestyle": ":",
    },
}

BAR_STYLES = {
    "primary": {
        "color": PALETTE["primary"],
        "edgecolor": PALETTE["primary_dark"],
        "linewidth": 0.8,
    },
    "baseline": {
        "color": PALETTE["neutral"],
        "edgecolor": "#555555",
        "linewidth": 0.8,
    },
    "emphasis": {
        "color": PALETTE["emphasis"],
        "edgecolor": "#CC7A00",
        "linewidth": 0.8,
    },
    "warning": {
        "color": PALETTE["warning"],
        "edgecolor": "#A64800",
        "linewidth": 0.8,
    },
}

FILL_STYLES = {
    "primary": {
        "color": PALETTE["primary"],
        "alpha": FIGURE_STANDARDS["alpha_fill"],
    },
    "light": {
        "color": PALETTE["light_gray"],
        "alpha": 0.25,
    },
    "zone": {
        "color": PALETTE["primary"],
        "alpha": FIGURE_STANDARDS["alpha_zone"],
    },
}

MARKER_STYLES = {
    "recommended": {
        "marker": "o",
        "s": 60,
        "c": PALETTE["emphasis"],
        "edgecolors": "black",
        "linewidths": 0.6,
        "zorder": 10,
    },
    "data_point": {
        "marker": "o",
        "s": 40,
        "c": PALETTE["primary"],
        "alpha": FIGURE_STANDARDS["alpha_scatter"],
    },
    "pareto": {
        "marker": "*",
        "s": 120,
        "c": PALETTE["emphasis"],
        "edgecolors": "black",
        "linewidths": 0.6,
        "zorder": 10,
    },
}

# ============================================================
# 工具函数
# ============================================================


def get_season_color(season: int) -> str:
    """根据赛季返回对应颜色（S1-2, S28+用灰; S3-27用蓝）"""
    if season <= 2 or season >= 28:
        return PALETTE["neutral"]
    return PALETTE["primary"]


def get_season_colors(seasons: list) -> list:
    """批量获取赛季颜色"""
    return [get_season_color(s) for s in seasons]


def get_method_color(method: str) -> str:
    """根据投票方法返回颜色"""
    if method.lower() in ["rank", "ranking"]:
        return PALETTE["neutral"]
    if method.lower() in ["percent", "percentage"]:
        return PALETTE["primary"]
    return PALETTE["primary_dark"]


def apply_paper_style(ax, grid_alpha: Optional[float] = None, title: Optional[str] = None):
    """
    为 matplotlib Axes 应用论文风格 (v1.0)
    
    参数:
        ax: matplotlib Axes 对象
        grid_alpha: 网格透明度（默认使用标准值）
        title: 可选的图内标题
    """
    if grid_alpha is None:
        grid_alpha = FIGURE_STANDARDS["alpha_grid"]

    # 只保留左、下轴
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_linewidth(FIGURE_STANDARDS["linewidth_border"])
    ax.spines["bottom"].set_linewidth(FIGURE_STANDARDS["linewidth_border"])

    # 刻度设置
    ax.tick_params(
        colors="#333333",
        labelsize=FIGURE_STANDARDS["tick_fontsize"],
        direction=FIGURE_STANDARDS["tick_direction"],
        length=FIGURE_STANDARDS["tick_major_size"],
    )

    # 网格设置
    ax.grid(
        True,
        alpha=grid_alpha,
        linestyle="--",
        linewidth=FIGURE_STANDARDS["linewidth_grid"],
        color=PALETTE["light_gray"],
    )

    if title:
        ax.set_title(
            title,
            fontsize=FIGURE_STANDARDS["title_fontsize"],
            fontweight="bold",
        )


def create_legend_patches():
    """创建标准图例 patches"""
    import matplotlib.patches as mpatches

    patches = {
        "rank": mpatches.Patch(color=PALETTE["neutral"], label=LEGEND_LABELS["rank"]),
        "percent": mpatches.Patch(color=PALETTE["primary"], label=LEGEND_LABELS["percent"]),
        "mismatch": mpatches.Patch(color=PALETTE["warning"], label=LEGEND_LABELS["mismatch"]),
        "recommended": mpatches.Patch(color=PALETTE["emphasis"], label=LEGEND_LABELS["recommended"]),
    }
    return patches


def add_confidence_band(ax, x, y_lower, y_upper, color=None, alpha=None, label=None):
    """
    添加置信区间/HDI 透明带
    
    参数:
        ax: matplotlib Axes
        x: x 坐标数组
        y_lower: 下界数组
        y_upper: 上界数组
        color: 填充颜色（默认使用主色）
        alpha: 透明度（默认使用标准值）
        label: 可选标签
    """
    if color is None:
        color = PALETTE["primary"]
    if alpha is None:
        alpha = FIGURE_STANDARDS["alpha_fill"]

    return ax.fill_between(x, y_lower, y_upper, color=color, alpha=alpha, label=label)


def add_highlight_zone(ax, x_min, x_max, label=None, color=None, alpha=None):
    """
    添加高亮区域（如 balanced zone）
    
    参数:
        ax: matplotlib Axes
        x_min, x_max: 区域范围
        label: 可选标签
        color: 填充颜色
        alpha: 透明度
    """
    if color is None:
        color = PALETTE["light_gray"]
    if alpha is None:
        alpha = FIGURE_STANDARDS["alpha_zone"]

    return ax.axvspan(x_min, x_max, color=color, alpha=alpha, label=label)


def add_recommended_point(ax, x, y, label="Recommended"):
    """
    添加推荐点标记（橙色 + 黑色描边）
    """
    return ax.scatter(
        x, y,
        **MARKER_STYLES["recommended"],
        label=label,
    )


# ============================================================
# LaTeX 颜色定义
# ============================================================

LATEX_COLORS = r"""
% DWTS Paper Palette v1.0 - LaTeX Color Definitions
% Add to main.tex preamble

\definecolor{dwts-primary}{HTML}{0072B2}      % 蓝 - 主结果
\definecolor{dwts-primary-dark}{HTML}{0B3C5D} % 深蓝 - 强调
\definecolor{dwts-emphasis}{HTML}{E69F00}     % 橙 - 推荐/DAWS
\definecolor{dwts-warning}{HTML}{D55E00}      % 红 - 风险/异常
\definecolor{dwts-neutral}{HTML}{7A7A7A}      % 灰 - 基线
\definecolor{dwts-light}{HTML}{D9D9D9}        % 浅灰 - 填充

% Backward compatibility aliases
\definecolor{dwts-proposed}{HTML}{0072B2}
\definecolor{dwts-baseline}{HTML}{7A7A7A}
\definecolor{dwts-accent}{HTML}{E69F00}

% tcolorbox styles
\newtcolorbox{proposedbox}[1][] {
    colback=dwts-primary!10!white,
    colframe=dwts-primary!80!black,
    #1
}
\newtcolorbox{warningbox}[1][] {
    colback=dwts-warning!10!white,
    colframe=dwts-warning!80!black,
    #1
}
\newtcolorbox{emphasisbox}[1][] {
    colback=dwts-emphasis!10!white,
    colframe=dwts-emphasis!80!black,
    #1
}
"""


if __name__ == "__main__":
    print("=" * 60)
    print("DWTS Paper Palette v1.0 - 论文级配色方案")
    print("=" * 60)
    print()
    print("语义色板:")
    for name in ["primary", "primary_dark", "emphasis", "warning", "neutral", "light_gray"]:
        color = PALETTE[name]
        print(f"  {name:14s}  {color}  {'█' * 10}")

    print()
    print("字体设置:")
    rcparams = get_paper_rcparams()
    print(f"  font.family: {rcparams['font.family']}")
    print(f"  font.serif:  {rcparams['font.serif']}")

    print()
    print("LaTeX 定义:")
    print(LATEX_COLORS)
