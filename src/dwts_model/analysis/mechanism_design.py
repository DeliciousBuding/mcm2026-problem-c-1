"""
机制设计与帕累托前沿分析。
以“评委一致性（技术公平）”与“观众一致性（参与度）”为双目标。
"""
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..paper_palette import PALETTE


@dataclass
class 机制设计结果:
    结果表: pd.DataFrame
    图像路径: str


def _format_alpha_label(min_alpha: float, max_alpha: float, count: int) -> str:
    if count <= 1 or np.isclose(min_alpha, max_alpha):
        return f"{min_alpha:.2f}"
    return f"{min_alpha:.2f}-{max_alpha:.2f}"


def _annotate_pareto_points(ax, xs, ys, labels):
    base_offsets = [(8, 8), (8, -8), (8, 12), (8, -12), (8, 16), (8, -16)]
    texts = []
    for idx, (x, y, label) in enumerate(zip(xs, ys, labels)):
        dx, dy = base_offsets[idx % len(base_offsets)]
        texts.append(
            ax.annotate(
                f"α={label}",
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.85, 
                          pad=0.3, boxstyle="round,pad=0.2"),
            )
        )

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for _ in range(60):
        moved = False
        bboxes = [t.get_window_extent(renderer=renderer).expanded(1.02, 1.2) for t in texts]
        for i in range(len(texts)):
            for j in range(i):
                if bboxes[i].overlaps(bboxes[j]):
                    dx, dy = texts[i].get_position()
                    shift = 6 if (i % 2 == 0) else -6
                    texts[i].set_position((dx, dy + shift))
                    moved = True
        if not moved:
            break
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


def _机制淘汰(fan_votes, judge_pct, alpha: float) -> str:
    combined = {k: alpha * judge_pct.get(k, 0.0) + (1 - alpha) * fan_votes.get(k, 0.0) for k in fan_votes}
    return min(combined.items(), key=lambda x: x[1])[0]


def 运行帕累托优化(
    manager,
    posterior_summary: pd.DataFrame,
    alphas: List[float],
    fig_path: str,
    dynamic_alpha_fn: Callable[[int, int], float] = None,
):
    records = []
    grouped = posterior_summary.groupby(["season", "week"])

    for alpha in alphas:
        judge_align = []
        fan_align = []
        for (season, week), group in grouped:
            ctx = manager.get_season_context(int(season))
            week_ctx = ctx.weeks.get(int(week))
            if week_ctx is None or not week_ctx.has_valid_elimination():
                continue

            fan_votes = dict(zip(group["contestant"], group["fan_mean"]))
            judge_pct = week_ctx.judge_percentages

            fan_elim = min(fan_votes.items(), key=lambda x: x[1])[0]
            judge_elim = min(judge_pct.items(), key=lambda x: x[1])[0] if judge_pct else None
            use_alpha = dynamic_alpha_fn(week, ctx.num_weeks) if dynamic_alpha_fn else alpha
            mech_elim = _机制淘汰(fan_votes, judge_pct, use_alpha)

            if judge_elim:
                judge_align.append(1.0 if mech_elim == judge_elim else 0.0)
            fan_align.append(1.0 if mech_elim == fan_elim else 0.0)

        records.append(
            {
                "alpha": alpha,
                "judge_alignment": float(np.mean(judge_align)) if judge_align else np.nan,
                "fan_influence": float(np.mean(fan_align)) if fan_align else np.nan,
                "wrongful_rate": float(1.0 - np.mean(fan_align)) if fan_align else np.nan,
            }
        )

    df = pd.DataFrame(records)

    # === 帕累托前沿图（高信息密度版本） ===
    from ..paper_palette import FIGURE_STANDARDS, apply_paper_style
    
    fig, ax = plt.subplots(figsize=FIGURE_STANDARDS["figsize_standard"])
    
    plot_df = df.dropna(subset=["judge_alignment", "fan_influence"]).copy()
    x_vals = plot_df["judge_alignment"].to_numpy()
    y_vals = plot_df["fan_influence"].to_numpy()
    
    # 帕累托区域填充（增加信息密度）
    sorted_idx = np.argsort(x_vals)
    x_sorted, y_sorted = x_vals[sorted_idx], y_vals[sorted_idx]
    ax.fill_between(x_sorted, y_sorted, y_sorted.min() - 0.005, 
                    alpha=FIGURE_STANDARDS["alpha_fill"], color=PALETTE["light_gray"],
                    label="Feasible Trade-off Region")
    
    # 主曲线（加粗）
    ax.plot(x_sorted, y_sorted, "o-",
            color=PALETTE["primary"],
            linewidth=FIGURE_STANDARDS["linewidth_main"],
            markersize=10, markeredgecolor=PALETTE["neutral"], markeredgewidth=1.5,
            label="Pareto Frontier", zorder=4)
    
    # 高亮推荐点（DAWS 区域：alpha 0.55-0.65）使用橙色强调
    daws_mask = (plot_df["alpha"] >= 0.55) & (plot_df["alpha"] <= 0.65)
    if daws_mask.any():
        daws_x = plot_df.loc[daws_mask, "judge_alignment"].values
        daws_y = plot_df.loc[daws_mask, "fan_influence"].values
        ax.scatter(daws_x, daws_y, s=FIGURE_STANDARDS["marker_size_large"] * 2.4,
                   c=PALETTE["emphasis"], edgecolor=PALETTE["primary_dark"],
                   linewidth=2.2, zorder=5, marker="*",
                   label="DAWS Recommended (α≈0.6)")
    
    # 标注alpha值
    grouped = plot_df.groupby(["judge_alignment", "fan_influence"], sort=False)
    label_df = grouped.agg(
        alpha_min=("alpha", "min"),
        alpha_max=("alpha", "max"),
        alpha_count=("alpha", "count"),
    ).reset_index()
    label_df["alpha_label"] = [
        _format_alpha_label(min_a, max_a, count)
        for min_a, max_a, count in zip(
            label_df["alpha_min"], label_df["alpha_max"], label_df["alpha_count"]
        )
    ]

    _annotate_pareto_points(
        ax,
        label_df["judge_alignment"].to_numpy(),
        label_df["fan_influence"].to_numpy(),
        label_df["alpha_label"].tolist(),
    )

    # 应用论文风格
    apply_paper_style(ax)
    ax.set_xlabel("Judge Alignment (Fairness)", fontsize=FIGURE_STANDARDS["label_fontsize"])
    ax.set_ylabel("Fan Influence Index", fontsize=FIGURE_STANDARDS["label_fontsize"])
    ax.set_title("DAWS lies on the most balanced Pareto segment",
                 fontsize=FIGURE_STANDARDS["title_fontsize"], fontweight='bold')
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, 
              fontsize=FIGURE_STANDARDS["legend_fontsize"])
    
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return 机制设计结果(结果表=df, 图像路径=fig_path)
