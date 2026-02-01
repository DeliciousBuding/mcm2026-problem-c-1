"""
DAWS (Dynamic Adaptive Weighting System) utilities for Q4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class DAWSResult:
    alpha_schedule: List[float]
    weekly_results: pd.DataFrame
    robustness: pd.DataFrame


def compute_alpha_schedule(
    total_weeks: int,
    uncertainty: Optional[List[float]] = None,
    alpha0: float = 0.6,
    gamma: float = -0.1,
    eta: float = 0.2,
    alpha_min: float = 0.35,
    alpha_max: float = 0.75,
    delta: float = 0.07,
) -> List[float]:
    if total_weeks <= 0:
        return []
    if uncertainty is None or len(uncertainty) != total_weeks:
        uncertainty = [0.0] * total_weeks
    u = np.array(uncertainty, dtype=float)
    if np.nanmax(u) > 0:
        u = (u - np.nanmin(u)) / max(np.nanmax(u) - np.nanmin(u), 1e-8)

    alphas: List[float] = []
    for t in range(1, total_weeks + 1):
        raw = alpha0 + gamma * (t / total_weeks) - eta * u[t - 1]
        alpha = float(np.clip(raw, alpha_min, alpha_max))
        if alphas:
            prev = alphas[-1]
            if abs(alpha - prev) > delta:
                alpha = prev + np.sign(alpha - prev) * delta
        alphas.append(alpha)
    return alphas


def _mechanism_elim(fan_votes: Dict[str, float], judge_pct: Dict[str, float], alpha: float) -> str:
    combined = {k: alpha * judge_pct.get(k, 0.0) + (1 - alpha) * fan_votes.get(k, 0.0) for k in fan_votes}
    return min(combined.items(), key=lambda x: x[1])[0]


def run_daws_weekly(manager, posterior_summary: pd.DataFrame, alpha_schedule: List[float]) -> pd.DataFrame:
    records = []
    grouped = posterior_summary.groupby(["season", "week"])
    for (season, week), group in grouped:
        ctx = manager.get_season_context(int(season))
        week_ctx = ctx.weeks.get(int(week))
        if week_ctx is None or not week_ctx.has_valid_elimination():
            continue
        alpha = alpha_schedule[min(week - 1, len(alpha_schedule) - 1)] if alpha_schedule else 0.6
        fan_votes = dict(zip(group["contestant"], group["fan_mean"]))
        judge_pct = week_ctx.judge_percentages
        elim = _mechanism_elim(fan_votes, judge_pct, alpha)
        records.append({"season": season, "week": week, "alpha": alpha, "elim": elim})
    return pd.DataFrame(records)


def run_noise_robustness(
    manager,
    posterior_summary: pd.DataFrame,
    alpha_schedule: List[float],
    noise_level: float = 0.05,
    trials: int = 200,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    baseline = run_daws_weekly(manager, posterior_summary, alpha_schedule)
    baseline_map = {(int(r.season), int(r.week)): r.elim for r in baseline.itertuples(index=False)}

    grouped = posterior_summary.groupby(["season", "week"])
    records = []
    for (season, week), group in grouped:
        ctx = manager.get_season_context(int(season))
        week_ctx = ctx.weeks.get(int(week))
        if week_ctx is None or not week_ctx.has_valid_elimination():
            continue
        alpha = alpha_schedule[min(week - 1, len(alpha_schedule) - 1)] if alpha_schedule else 0.6
        fan_votes = dict(zip(group["contestant"], group["fan_mean"]))
        judge_pct = week_ctx.judge_percentages
        baseline_elim = baseline_map.get((int(season), int(week)))

        flip = 0
        for _ in range(trials):
            noisy_fan = {k: max(0.0, v * (1 + rng.normal(0, noise_level))) for k, v in fan_votes.items()}
            noisy_judge = {k: max(0.0, v * (1 + rng.normal(0, noise_level))) for k, v in judge_pct.items()}
            # normalize
            fsum = sum(noisy_fan.values())
            jsum = sum(noisy_judge.values())
            if fsum > 0:
                noisy_fan = {k: v / fsum for k, v in noisy_fan.items()}
            if jsum > 0:
                noisy_judge = {k: v / jsum for k, v in noisy_judge.items()}
            elim = _mechanism_elim(noisy_fan, noisy_judge, alpha)
            if elim != baseline_elim:
                flip += 1

        records.append({
            "season": season,
            "week": week,
            "flip_rate": flip / trials,
        })

    return pd.DataFrame(records)


def plot_alpha_schedule(alpha_schedule: List[float], fig_path: str):
    """绘制DAWS权重曲线（高信息密度版本）"""
    if not alpha_schedule:
        return
    
    from ..paper_palette import PALETTE, FIGURE_STANDARDS, apply_paper_style
    
    weeks = list(range(1, len(alpha_schedule) + 1))
    fig, ax = plt.subplots(figsize=FIGURE_STANDARDS["figsize_standard"])
    
    # 填充背景区域（显示alpha范围）
    ax.fill_between(weeks, 0.35, 0.75, alpha=0.15, color=PALETTE["light_gray"], 
                    label="Valid α Range [0.35, 0.75]")
    ax.axhline(y=0.6, color=PALETTE["neutral"], linestyle="--", 
               linewidth=1.5, alpha=0.6, label="Baseline α=0.6")
    
    # 主曲线（深蓝 + 金色强调点）
    ax.plot(weeks, alpha_schedule, color=PALETTE["primary"],
            linewidth=FIGURE_STANDARDS["linewidth_main"], zorder=3, label="DAWS alpha(t)")
    ax.scatter(weeks, alpha_schedule, color=PALETTE["primary"],
               s=FIGURE_STANDARDS["marker_size_small"], edgecolor=PALETTE["primary_dark"],
               linewidth=0.8, zorder=4)

    highlight_idx = int(np.argmin(np.abs(np.array(alpha_schedule) - 0.6)))
    ax.scatter(weeks[highlight_idx], alpha_schedule[highlight_idx], color=PALETTE["emphasis"],
               s=FIGURE_STANDARDS["marker_size_large"] * 1.15, edgecolor=PALETTE["primary_dark"],
               linewidth=1.6, zorder=5, label="DAWS Recommended")

    # 标注起止点
    ax.annotate(f"α₀={alpha_schedule[0]:.2f}", xy=(1, alpha_schedule[0]),
                xytext=(10, 10), textcoords="offset points", fontsize=10,
                fontweight='bold', bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    ax.annotate(f"αₜ={alpha_schedule[-1]:.2f}", xy=(len(alpha_schedule), alpha_schedule[-1]),
                xytext=(10, -10), textcoords="offset points", fontsize=10,
                fontweight='bold', bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    
    apply_paper_style(ax)
    ax.set_xlabel("Week", fontsize=FIGURE_STANDARDS["label_fontsize"])
    ax.set_ylabel("Judge Weight (α)", fontsize=FIGURE_STANDARDS["label_fontsize"])
    ax.set_title("DAWS keeps judge weight within valid bounds", fontsize=FIGURE_STANDARDS["title_fontsize"], fontweight='bold')
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=FIGURE_STANDARDS["legend_fontsize"])
    ax.set_ylim(0.3, 0.8)
    
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_noise_robustness(robust_df: pd.DataFrame, fig_path: str):
    """绘制噪声鲁棒性图（高信息密度版本）"""
    if robust_df.empty:
        return
    
    from ..paper_palette import PALETTE, FIGURE_STANDARDS, apply_paper_style
    
    summary = robust_df.groupby("season")["flip_rate"].mean().reset_index()
    fig, ax = plt.subplots(figsize=FIGURE_STANDARDS["figsize_standard"])
    
    x = summary["season"].to_numpy()
    y = summary["flip_rate"].to_numpy()
    
    # Lollipop 图（蓝灰配色）
    for xi, yi in zip(x, y):
        ax.plot([xi, xi], [0, yi], color=PALETTE["light_gray"], 
                linewidth=FIGURE_STANDARDS["linewidth_secondary"], alpha=0.85)
    ax.scatter(x, y, color=PALETTE["primary"], edgecolor=PALETTE["primary_dark"], 
               s=FIGURE_STANDARDS["marker_size"], linewidth=1.5, zorder=3)
    
    # 添加均值参考线
    mean_flip = np.mean(y)
    ax.axhline(y=mean_flip, color=PALETTE["neutral"], linestyle="--", 
               linewidth=1.5, alpha=0.7, label=f"Mean: {mean_flip:.3f}")
    
    apply_paper_style(ax)
    ax.set_xlabel("Season", fontsize=FIGURE_STANDARDS["label_fontsize"])
    ax.set_ylabel("Flip Rate under Noise", fontsize=FIGURE_STANDARDS["label_fontsize"])
    ax.set_title("DAWS remains stable under small noise", fontsize=FIGURE_STANDARDS["title_fontsize"], fontweight='bold')
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=FIGURE_STANDARDS["legend_fontsize"])
    
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
