"""
DWTS 图表模板模块 (v1.0)

提供各类图表的标准化模板函数，确保全篇图表风格统一。
所有模板自动应用 paper_palette 中的配色和样式规范。

模板类型：
- 折线图模板 (Fig.9/12/14 类)
- 热力图模板 (Fig.4/6 类)
- 棒棒糖图模板 (Fig.8 类)
- Pareto前沿图模板 (Fig.13 左)
- SHAP解释图模板 (Fig.10/11 类)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Tuple, Union
from pathlib import Path

from .paper_palette import (
    PALETTE, FIGURE_STANDARDS, 
    get_paper_rcparams, apply_paper_style,
    LINE_STYLES, MARKER_STYLES,
    add_confidence_band, add_recommended_point,
)


def setup_paper_style():
    """
    初始化论文级绘图样式。
    在脚本开头调用一次即可。
    """
    plt.rcParams.update(get_paper_rcparams())


# ============================================================
# 折线图模板 (Fig.9/12/14 类)
# ============================================================

def create_line_chart(
    figsize: Tuple[float, float] = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    创建折线图基础模板。
    
    模板规则：
    - 主线：蓝色
    - 推荐点：橙色 + 黑色细描边
    - 基线：灰色虚线
    - 不确定性：同色透明带
    
    返回:
        (fig, ax) 元组
    """
    if figsize is None:
        figsize = FIGURE_STANDARDS["figsize_standard"]
    
    fig, ax = plt.subplots(figsize=figsize)
    apply_paper_style(ax, title=title)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FIGURE_STANDARDS["label_fontsize"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FIGURE_STANDARDS["label_fontsize"])
    
    return fig, ax


def plot_main_line(ax, x, y, label: str = None, with_uncertainty: bool = False,
                   y_lower=None, y_upper=None):
    """
    绑定主线（蓝色实线）+ 可选置信区间。
    
    参数:
        ax: Axes 对象
        x: x 坐标
        y: y 坐标
        label: 图例标签
        with_uncertainty: 是否添加置信带
        y_lower, y_upper: 置信区间上下界
    """
    line, = ax.plot(x, y, **LINE_STYLES["primary"], label=label)
    
    if with_uncertainty and y_lower is not None and y_upper is not None:
        add_confidence_band(ax, x, y_lower, y_upper, color=PALETTE["primary"])
    
    return line


def plot_baseline(ax, x, y, label: str = "Baseline"):
    """绘制基线（灰色虚线）"""
    return ax.plot(x, y, **LINE_STYLES["baseline"], label=label)


def plot_recommended_points(ax, x, y, label: str = "Recommended"):
    """标记推荐点（橙色 + 黑色描边）"""
    return add_recommended_point(ax, x, y, label=label)


# ============================================================
# 热力图模板 (Fig.4/6 类)
# ============================================================

def create_heatmap(
    data: np.ndarray,
    row_labels: List[str] = None,
    col_labels: List[str] = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    cbar_label: str = None,
    figsize: Tuple[float, float] = None,
    cmap: str = "cividis",
    vmin: float = None,
    vmax: float = None,
    annot: bool = False,
    fmt: str = ".2f",
    highlight_cells: List[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    创建热力图模板。
    
    模板规则：
    - 顺序色带：cividis 或蓝色渐变
    - colorbar 右侧，标签写清楚
    - 可选圈选重点（橙色细圆圈）
    
    参数:
        data: 2D numpy 数组
        row_labels, col_labels: 行列标签
        cbar_label: colorbar 标签（变量名 + 单位）
        highlight_cells: 要圈选的单元格坐标列表 [(row, col), ...]
    """
    if figsize is None:
        # 根据数据维度自动调整
        aspect = data.shape[1] / data.shape[0]
        width = min(8, max(5, data.shape[1] * 0.5))
        height = width / aspect
        figsize = (width, max(height, 4))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=FIGURE_STANDARDS["label_fontsize"])
    cbar.ax.tick_params(labelsize=FIGURE_STANDARDS["tick_fontsize"])
    
    # 标签
    if row_labels is not None:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=FIGURE_STANDARDS["tick_fontsize"])
    if col_labels is not None:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=FIGURE_STANDARDS["tick_fontsize"], 
                          rotation=45, ha="right")
    
    # 标注
    if annot:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                color = "white" if val > (data.max() + data.min()) / 2 else "black"
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                       fontsize=FIGURE_STANDARDS["annotation_fontsize"], color=color)
    
    # 圈选高亮（橙色细圆圈）
    if highlight_cells:
        for row, col in highlight_cells:
            rect = mpatches.Rectangle(
                (col - 0.5, row - 0.5), 1, 1,
                fill=False,
                edgecolor=PALETTE["emphasis"],
                linewidth=1.2,
            )
            ax.add_patch(rect)
    
    if title:
        ax.set_title(title, fontsize=FIGURE_STANDARDS["title_fontsize"], fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FIGURE_STANDARDS["label_fontsize"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FIGURE_STANDARDS["label_fontsize"])
    
    plt.tight_layout()
    return fig, ax


# ============================================================
# 棒棒糖图模板 (Fig.8 类)
# ============================================================

def create_lollipop_chart(
    values: np.ndarray,
    labels: List[str],
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: Tuple[float, float] = None,
    sort_by_value: bool = False,
    highlight_indices: List[int] = None,
    orientation: str = "horizontal",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    创建棒棒糖图模板。
    
    模板规则：
    - 点：蓝色
    - 竖线：浅蓝/浅灰
    - 若强调某项：浅灰背景带
    - 非时间序列：按值排序
    
    参数:
        values: 数值数组
        labels: 标签列表
        sort_by_value: 是否按值排序
        highlight_indices: 要高亮的索引（显示浅灰背景）
        orientation: "horizontal" 或 "vertical"
    """
    if figsize is None:
        if orientation == "horizontal":
            figsize = (6.4, max(3.5, len(values) * 0.25))
        else:
            figsize = FIGURE_STANDARDS["figsize_standard"]
    
    # 排序
    if sort_by_value:
        order = np.argsort(values)
        values = values[order]
        labels = [labels[i] for i in order]
        if highlight_indices:
            # 重新映射高亮索引
            index_map = {old: new for new, old in enumerate(order)}
            highlight_indices = [index_map.get(i, -1) for i in highlight_indices]
            highlight_indices = [i for i in highlight_indices if i >= 0]
    
    fig, ax = plt.subplots(figsize=figsize)
    apply_paper_style(ax, title=title)
    
    positions = np.arange(len(values))
    
    if orientation == "horizontal":
        # 高亮背景带
        if highlight_indices:
            for idx in highlight_indices:
                ax.axhspan(idx - 0.4, idx + 0.4, 
                          color=PALETTE["light_gray"], 
                          alpha=FIGURE_STANDARDS["alpha_zone"])
        
        # 茎线
        ax.hlines(positions, 0, values, 
                 color=PALETTE["light_gray"], 
                 linewidth=FIGURE_STANDARDS["linewidth_baseline"])
        
        # 点
        ax.scatter(values, positions, 
                  color=PALETTE["primary"], 
                  s=FIGURE_STANDARDS["marker_size"] * 15,
                  zorder=5)
        
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=FIGURE_STANDARDS["label_fontsize"])
    else:
        # 高亮背景带
        if highlight_indices:
            for idx in highlight_indices:
                ax.axvspan(idx - 0.4, idx + 0.4,
                          color=PALETTE["light_gray"],
                          alpha=FIGURE_STANDARDS["alpha_zone"])
        
        # 茎线
        ax.vlines(positions, 0, values,
                 color=PALETTE["light_gray"],
                 linewidth=FIGURE_STANDARDS["linewidth_baseline"])
        
        # 点
        ax.scatter(positions, values,
                  color=PALETTE["primary"],
                  s=FIGURE_STANDARDS["marker_size"] * 15,
                  zorder=5)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=FIGURE_STANDARDS["label_fontsize"])
    
    plt.tight_layout()
    return fig, ax


# ============================================================
# Pareto 前沿图模板 (Fig.13 左)
# ============================================================

def create_pareto_chart(
    x: np.ndarray,
    y: np.ndarray,
    pareto_x: np.ndarray = None,
    pareto_y: np.ndarray = None,
    recommended_idx: int = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: Tuple[float, float] = None,
    show_feasible: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    创建 Pareto 前沿图模板。
    
    模板规则：
    - 可行域：浅蓝透明填充
    - Pareto frontier：深蓝线
    - 推荐点：橙色星形标记
    - 其它点：蓝色散点
    
    参数:
        x, y: 所有数据点坐标
        pareto_x, pareto_y: Pareto 前沿点坐标
        recommended_idx: 推荐点在原数组中的索引
    """
    if figsize is None:
        figsize = FIGURE_STANDARDS["figsize_square"]
    
    fig, ax = plt.subplots(figsize=figsize)
    apply_paper_style(ax, title=title)
    
    # 可行域填充
    if show_feasible:
        hull_x = np.concatenate([[x.min()], pareto_x, [x.max()]])
        hull_y = np.concatenate([[y.max()], pareto_y, [y.max()]])
        ax.fill(hull_x, hull_y, 
               color=PALETTE["primary"], 
               alpha=FIGURE_STANDARDS["alpha_zone"],
               label="Feasible Region")
    
    # 所有数据点
    ax.scatter(x, y, 
              color=PALETTE["primary"],
              alpha=FIGURE_STANDARDS["alpha_scatter"],
              s=FIGURE_STANDARDS["marker_size"] * 8,
              label="Solutions")
    
    # Pareto 前沿线
    if pareto_x is not None and pareto_y is not None:
        ax.plot(pareto_x, pareto_y,
               color=PALETTE["primary_dark"],
               linewidth=FIGURE_STANDARDS["linewidth_main"],
               marker="o",
               markersize=FIGURE_STANDARDS["marker_size"],
               label="Pareto Frontier")
    
    # 推荐点（橙色星形）
    if recommended_idx is not None:
        ax.scatter(x[recommended_idx], y[recommended_idx],
                  **MARKER_STYLES["pareto"],
                  label="Recommended")
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FIGURE_STANDARDS["label_fontsize"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FIGURE_STANDARDS["label_fontsize"])
    
    ax.legend(loc="best", fontsize=FIGURE_STANDARDS["legend_fontsize"])
    
    plt.tight_layout()
    return fig, ax


# ============================================================
# 保存工具
# ============================================================

def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    formats: List[str] = None,
    dpi: int = 300,
):
    """
    保存图表（自动处理多格式）。
    
    规则：
    - 线图/矢量图：优先 PDF/SVG
    - 栅格图：至少 300 dpi
    
    参数:
        fig: Figure 对象
        path: 保存路径（不含扩展名）
        formats: 格式列表，默认 ["pdf", "png"]
        dpi: 栅格图 dpi
    """
    if formats is None:
        formats = ["pdf", "png"]
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        save_path = path.with_suffix(f".{fmt}")
        fig.savefig(
            save_path,
            format=fmt,
            dpi=dpi if fmt in ["png", "jpg", "jpeg"] else None,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    
    plt.close(fig)


# ============================================================
# SHAP 解释图模板 (Fig.10/11 类)
# ============================================================

def create_shap_importance_bar(
    feature_names: List[str],
    importance_values: np.ndarray,
    title: str = "Feature Importance (mean |SHAP|)",
    xlabel: str = "Mean |SHAP value|",
    figsize: Tuple[float, float] = None,
    max_display: int = 15,
    abbreviate_names: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    创建 SHAP 特征重要性条形图模板。
    
    模板规则：
    - 特征名统一缩写规则
    - 顺序色带或单色
    - 主文只放 Top 10/15
    
    参数:
        feature_names: 特征名列表
        importance_values: 重要性值数组（mean |SHAP|）
        max_display: 最多显示多少个特征
        abbreviate_names: 是否缩写特征名
    """
    if figsize is None:
        n_features = min(len(feature_names), max_display)
        figsize = (6.4, max(3.5, n_features * 0.28))
    
    # 排序并截取
    order = np.argsort(importance_values)[::-1][:max_display]
    values = importance_values[order]
    names = [feature_names[i] for i in order]
    
    # 缩写特征名
    if abbreviate_names:
        names = [_abbreviate_feature_name(n) for n in names]
    
    fig, ax = plt.subplots(figsize=figsize)
    apply_paper_style(ax)
    
    positions = np.arange(len(values))
    
    # 水平条形图（蓝色）
    ax.barh(
        positions, values,
        color=PALETTE["primary"],
        edgecolor=PALETTE["primary_dark"],
        linewidth=0.6,
        alpha=0.85,
    )
    
    ax.set_yticks(positions)
    ax.set_yticklabels(names[::-1], fontsize=FIGURE_STANDARDS["tick_fontsize"])
    ax.set_xlabel(xlabel, fontsize=FIGURE_STANDARDS["label_fontsize"])
    if title:
        ax.set_title(title, fontsize=FIGURE_STANDARDS["title_fontsize"], fontweight="bold")
    
    ax.invert_yaxis()  # 最重要的在上面
    
    plt.tight_layout()
    return fig, ax


def create_shap_diverging_bar(
    feature_names: List[str],
    shap_values: np.ndarray,
    title: str = "SHAP Values",
    xlabel: str = "SHAP value (impact on output)",
    figsize: Tuple[float, float] = None,
    max_display: int = 15,
    abbreviate_names: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    创建 SHAP 发散条形图模板（正负影响）。
    
    模板规则：
    - 发散色带以 0 为中心对称
    - 正值用蓝色（有利），负值用红色（风险）
    
    参数:
        feature_names: 特征名列表
        shap_values: SHAP 值数组（可正可负）
        max_display: 最多显示多少个特征
    """
    if figsize is None:
        n_features = min(len(feature_names), max_display)
        figsize = (6.4, max(3.5, n_features * 0.28))
    
    # 按绝对值排序
    order = np.argsort(np.abs(shap_values))[::-1][:max_display]
    values = shap_values[order]
    names = [feature_names[i] for i in order]
    
    if abbreviate_names:
        names = [_abbreviate_feature_name(n) for n in names]
    
    fig, ax = plt.subplots(figsize=figsize)
    apply_paper_style(ax)
    
    positions = np.arange(len(values))
    
    # 正负不同颜色
    colors = [PALETTE["primary"] if v >= 0 else PALETTE["warning"] for v in values]
    
    ax.barh(
        positions, values,
        color=colors,
        edgecolor=[PALETTE["primary_dark"] if v >= 0 else "#A64800" for v in values],
        linewidth=0.6,
        alpha=0.85,
    )
    
    # 零线
    ax.axvline(0, color=PALETTE["neutral"], linestyle="-", linewidth=0.8)
    
    ax.set_yticks(positions)
    ax.set_yticklabels(names[::-1], fontsize=FIGURE_STANDARDS["tick_fontsize"])
    ax.set_xlabel(xlabel, fontsize=FIGURE_STANDARDS["label_fontsize"])
    if title:
        ax.set_title(title, fontsize=FIGURE_STANDARDS["title_fontsize"], fontweight="bold")
    
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig, ax


def _abbreviate_feature_name(name: str) -> str:
    """
    缩写特征名（例如 celebrity_industry_* → Industry: *）
    """
    # 常见缩写规则
    abbrevs = {
        "celebrity_industry_": "Industry: ",
        "celebrity_age": "Age",
        "ballroom_partner": "Pro Dancer",
        "total_score": "Score",
        "fan_vote_estimate": "Fan Vote",
        "elimination_week": "Elim. Week",
        "placement": "Placement",
        "prior_dance_exp": "Dance Exp.",
        "is_athlete": "Athlete",
        "is_musician": "Musician",
        "is_actor": "Actor",
    }
    
    for prefix, replacement in abbrevs.items():
        if name.startswith(prefix):
            return replacement + name[len(prefix):].replace("_", " ").title()
        if name == prefix.rstrip("_"):
            return replacement
    
    # 默认：下划线转空格，首字母大写
    return name.replace("_", " ").title()


# ============================================================
# 条形图模板 (对比图)
# ============================================================

def create_comparison_bar(
    categories: List[str],
    values_dict: dict,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: Tuple[float, float] = None,
    orientation: str = "vertical",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    创建对比条形图模板。
    
    参数:
        categories: 类别标签列表
        values_dict: {"组名": [值列表]} 字典
        orientation: "vertical" 或 "horizontal"
    
    用法:
        create_comparison_bar(
            categories=["Method A", "Method B"],
            values_dict={"Accuracy": [0.85, 0.92], "F1": [0.80, 0.88]}
        )
    """
    if figsize is None:
        figsize = FIGURE_STANDARDS["figsize_standard"]
    
    n_categories = len(categories)
    n_groups = len(values_dict)
    
    # 颜色列表（按语义）
    color_list = [
        PALETTE["primary"],
        PALETTE["emphasis"],
        PALETTE["neutral"],
        PALETTE["primary_dark"],
    ]
    
    fig, ax = plt.subplots(figsize=figsize)
    apply_paper_style(ax)
    
    bar_width = 0.8 / n_groups
    positions = np.arange(n_categories)
    
    for i, (group_name, group_values) in enumerate(values_dict.items()):
        offset = (i - n_groups / 2 + 0.5) * bar_width
        color = color_list[i % len(color_list)]
        
        if orientation == "vertical":
            ax.bar(
                positions + offset, group_values,
                width=bar_width,
                label=group_name,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
        else:
            ax.barh(
                positions + offset, group_values,
                height=bar_width,
                label=group_name,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
    
    if orientation == "vertical":
        ax.set_xticks(positions)
        ax.set_xticklabels(categories, fontsize=FIGURE_STANDARDS["tick_fontsize"])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=FIGURE_STANDARDS["label_fontsize"])
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels(categories, fontsize=FIGURE_STANDARDS["tick_fontsize"])
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=FIGURE_STANDARDS["label_fontsize"])
    
    if title:
        ax.set_title(title, fontsize=FIGURE_STANDARDS["title_fontsize"], fontweight="bold")
    
    ax.legend(loc="best", fontsize=FIGURE_STANDARDS["legend_fontsize"])
    
    plt.tight_layout()
    return fig, ax


# ============================================================
# 森林图模板 (Hazard Ratio / Effect Size)
# ============================================================

def create_forest_plot(
    labels: List[str],
    effects: np.ndarray,
    lower_ci: np.ndarray,
    upper_ci: np.ndarray,
    title: str = "Forest Plot",
    xlabel: str = "Effect Size",
    reference_line: float = 1.0,
    figsize: Tuple[float, float] = None,
    significant: List[bool] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    创建森林图模板（适用于 Hazard Ratio、Odds Ratio 等）。
    
    模板规则：
    - 效应点 + 置信区间误差线
    - 参考线（通常为 1）
    - 显著效应用橙色标记
    
    参数:
        labels: 变量名列表
        effects: 效应值数组（如 HR）
        lower_ci, upper_ci: 置信区间下上界
        reference_line: 参考线位置（HR=1, 或 β=0）
        significant: 是否显著的布尔列表
    """
    if figsize is None:
        figsize = (6.4, max(4, len(labels) * 0.35))
    
    fig, ax = plt.subplots(figsize=figsize)
    apply_paper_style(ax)
    
    positions = np.arange(len(labels))
    
    # 参考线
    ax.axvline(reference_line, color=PALETTE["neutral"], linestyle="--", linewidth=1.0, alpha=0.7)
    
    # 颜色（显著用橙色）
    if significant is None:
        colors = [PALETTE["primary"]] * len(labels)
    else:
        colors = [PALETTE["emphasis"] if s else PALETTE["primary"] for s in significant]
    
    for i, (label, effect, low, up, color) in enumerate(zip(labels, effects, lower_ci, upper_ci, colors)):
        ax.errorbar(
            effect, i,
            xerr=[[effect - low], [up - effect]],
            fmt="o",
            color=color,
            markersize=FIGURE_STANDARDS["marker_size"] * 1.5,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
        )
    
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=FIGURE_STANDARDS["tick_fontsize"])
    ax.set_xlabel(xlabel, fontsize=FIGURE_STANDARDS["label_fontsize"])
    if title:
        ax.set_title(title, fontsize=FIGURE_STANDARDS["title_fontsize"], fontweight="bold")
    
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig, ax

