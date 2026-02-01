# -*- coding: utf-8 -*-
"""
One-click pipeline for DWTS Problem C:
A) Hard-constraint inversion (LP/MILP)
B) MaxEnt + Bayesian smoothing (Hit-and-Run + RW prior)
C) PPC + risk diagnostics
D) Counterfactuals + information-theory deficit
E) ML/XAI (forward-chaining XGB + Cox + SHAP)
F) DAWS mechanism design + robustness
G) Figures + paper sync + compile

注意：脚本位于 src/scripts/，PROJECT_ROOT 指向项目根目录
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 项目根目录（从 src/scripts 向上两级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data" / "2026_MCM_Problem_C_Data.csv"
PANEL_PATH = PROJECT_ROOT / "data" / "额外补充数据集" / "external_outputs" / "weekly_panel_enriched.csv"
# 图表统一输出到 paper/figures，供中英文论文共享
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

sys.path.insert(0, str(SRC_DIR))

from dwts_model.etl import DWTSDataLoader, ActiveSetManager
from dwts_model.engines import PercentLPEngine, MILPRankEngine, RuleEngine, JudgeSaveMode
from dwts_model.engines.rule_engine import rank_from_share
from dwts_model.sampling import (
    sample_week,
    summarize_posterior,
    hit_and_run_sample,
    importance_resample_with_prior,
    project_to_simplex_with_bounds,
)
from dwts_model.analysis import (
    run_pareto_frontier,
    run_ml_pipeline,
    compute_alpha_schedule,
    run_daws_weekly,
    run_noise_robustness,
    plot_alpha_schedule,
    plot_noise_robustness,
)

# 使用统一配色方案 v1.0
from dwts_model.paper_palette import PALETTE as _BASE_PALETTE, get_paper_rcparams

# 应用论文级样式
import matplotlib as mpl
mpl.rcParams.update(get_paper_rcparams())

# 兼容旧代码的颜色别名
PALETTE = {
    **_BASE_PALETTE,
    # 旧代码兼容别名（映射到新的语义色）
    "pale_blue": "#E7EFF3",
    "light_blue": _BASE_PALETTE["light_gray"],
    "cyan_blue": _BASE_PALETTE["primary"],
    "deep_blue": _BASE_PALETTE["primary_dark"],
    "navy": _BASE_PALETTE["primary_dark"],
    "arrow_gray": _BASE_PALETTE["neutral"],
    "text_gray": "#555555",
    "ref_gray": _BASE_PALETTE["neutral"],
    "daws_gold": _BASE_PALETTE["emphasis"],
    "accent": _BASE_PALETTE["emphasis"],
}


def _blue_gray_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "dwts_blue_gray",
        ["#E7EFF3", PALETTE["primary"], PALETTE["primary_dark"]],
        N=256,
    )


def _run_cmd(cmd, cwd: Path):
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _bounds_to_arrays(bounds: Dict[str, Tuple[float, float]]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    names = list(bounds.keys())
    lower = np.array([bounds[n][0] for n in names], dtype=float)
    upper = np.array([bounds[n][1] for n in names], dtype=float)
    return names, lower, upper


def _warm_start_from_prev(prev_mean: Optional[Dict[str, float]], bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    names, lower, upper = _bounds_to_arrays(bounds)
    if prev_mean:
        vec = np.array([prev_mean.get(n, (bounds[n][0] + bounds[n][1]) / 2) for n in names], dtype=float)
    else:
        vec = np.array([(bounds[n][0] + bounds[n][1]) / 2 for n in names], dtype=float)
    vec = project_to_simplex_with_bounds(vec, lower, upper)
    return {names[i]: float(vec[i]) for i in range(len(names))}


def _is_sample_feasible(rule_engine: RuleEngine, week_ctx, sample: Dict[str, float], method: str, has_judges_save: bool) -> bool:
    contestants = list(week_ctx.active_set)
    if not contestants:
        return False
    eliminated = set(week_ctx.eliminated)
    if method == "percent":
        res = rule_engine.percent_rule(sample, week_ctx.judge_percentages, alpha=0.5)
        return res.eliminated in eliminated

    fan_rank = {c: r for c, r in rank_from_share(sample).items()}
    res = rule_engine.rank_rule(fan_rank, week_ctx.judge_ranks)
    if has_judges_save:
        bottom_two = set(res.bottom_two or ())
        return any(e in bottom_two for e in eliminated)
    return res.eliminated in eliminated


def _gap_probability(rule_engine: RuleEngine, week_ctx, samples: List[Dict[str, float]], tau: float = 0.01) -> float:
    if not samples:
        return float("nan")
    gaps = []
    for s in samples:
        res = rule_engine.percent_rule(s, week_ctx.judge_percentages, alpha=0.5)
        combined = res.combined
        ordered = sorted(combined.values())
        if len(ordered) < 2:
            continue
        gap = ordered[1] - ordered[0]
        gaps.append(gap)
    if not gaps:
        return float("nan")
    gaps = np.array(gaps)
    return float(np.mean(gaps < tau))


def run_q1_inversion_sampling(
    manager: ActiveSetManager,
    n_samples: int = 2000,
    smooth_sigma: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    lp_engine = PercentLPEngine()
    rank_engine = MILPRankEngine()
    rule_engine = RuleEngine(seed=42)

    interval_records = []
    posterior_records = []
    accept_records = []
    gap_records = []
    samples_by_week: dict = {}

    rng = np.random.default_rng(42)

    for season in manager.get_all_seasons():
        ctx = manager.get_season_context(season)
        engine = lp_engine if ctx.voting_method == "percent" else rank_engine
        inversion = engine.solve(ctx)

        prev_mean: Optional[Dict[str, float]] = None
        for week, week_ctx in ctx.weeks.items():
            if not week_ctx.has_valid_elimination():
                continue

            bounds: Dict[str, Tuple[float, float]] = {}
            for contestant in week_ctx.active_set:
                est = inversion.week_results.get(week, {}).get(contestant)
                if est:
                    bounds[contestant] = (est.lower_bound, est.upper_bound)
                else:
                    bounds[contestant] = (0.01, 0.99)

                interval_records.append(
                    {
                        "season": season,
                        "week": week,
                        "contestant": contestant,
                        "voting_method": ctx.voting_method,
                        "lower": bounds[contestant][0],
                        "upper": bounds[contestant][1],
                        "point": est.point_estimate if est else np.nan,
                    }
                )

            start = _warm_start_from_prev(prev_mean, bounds)
            try:
                samples = hit_and_run_sample(bounds, n_samples=n_samples, start=start, burnin=200, thin=2, rng=rng)
            except Exception:
                samples = []

            if len(samples) < max(50, n_samples // 5):
                samples = sample_week(bounds, n_samples=n_samples, burnin=500, thin=5, smooth_lambda=10.0, prev_sample=prev_mean, rng=rng)

            samples = importance_resample_with_prior(samples, prev_mean, sigma=smooth_sigma, rng=rng)

            feasible = [s for s in samples if _is_sample_feasible(rule_engine, week_ctx, s, ctx.voting_method, ctx.has_judges_save)]
            accept_rate = len(feasible) / max(len(samples), 1)
            accept_records.append({"season": season, "week": week, "accept_rate": accept_rate})

            final_samples = feasible if feasible else samples
            summary = summarize_posterior(final_samples)
            for contestant, (mean_val, hdi_low, hdi_high) in summary.items():
                posterior_records.append(
                    {
                        "season": season,
                        "week": week,
                        "contestant": contestant,
                        "fan_mean": mean_val,
                        "fan_hdi_low": hdi_low,
                        "fan_hdi_high": hdi_high,
                    }
                )

            gap_prob = _gap_probability(rule_engine, week_ctx, final_samples, tau=0.01)
            gap_records.append({"season": season, "week": week, "gap_prob": gap_prob})

            samples_by_week[(season, week)] = final_samples
            prev_mean = {k: v[0] for k, v in summary.items()} if summary else None

    intervals_df = pd.DataFrame(interval_records)
    posterior_df = pd.DataFrame(posterior_records)
    accept_df = pd.DataFrame(accept_records)
    gap_df = pd.DataFrame(gap_records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    intervals_df.to_csv(OUTPUT_DIR / "fan_vote_intervals.csv", index=False)
    posterior_df.to_csv(OUTPUT_DIR / "fan_vote_posterior_summary.csv", index=False)
    accept_df.to_csv(OUTPUT_DIR / "accept_rate.csv", index=False)
    gap_df.to_csv(OUTPUT_DIR / "gap_probability.csv", index=False)

    return posterior_df, intervals_df, samples_by_week, accept_df, gap_df


def plot_hdi_band(manager: ActiveSetManager, posterior_df: pd.DataFrame, fig_path: Path):
    if posterior_df.empty:
        return
    season = posterior_df.groupby("season")["week"].nunique().sort_values(ascending=False).index[0]
    df = posterior_df[posterior_df["season"] == season]
    top = df.groupby("contestant")["fan_mean"].mean().sort_values(ascending=False).head(3).index.tolist()
    from dwts_model.paper_palette import FIGURE_STANDARDS, apply_paper_style
    fig, ax = plt.subplots(figsize=FIGURE_STANDARDS["figsize_standard"])
    line_styles = ["-", "--", ":"]
    elim_by_contestant = {}
    ctx = manager.get_season_context(int(season))
    for week, week_ctx in ctx.weeks.items():
        if not week_ctx.has_valid_elimination():
            continue
        for name in week_ctx.eliminated:
            elim_by_contestant.setdefault(name, []).append(int(week))

    for idx, name in enumerate(top):
        sub = df[df["contestant"] == name].sort_values("week")
        linestyle = line_styles[idx % len(line_styles)]
        is_eliminated = name in elim_by_contestant
        lw = FIGURE_STANDARDS["linewidth_main"] + (0.4 if is_eliminated else 0.0)
        ax.fill_between(sub["week"], sub["fan_hdi_low"], sub["fan_hdi_high"],
                        alpha=FIGURE_STANDARDS["alpha_fill"], color=PALETTE["light_blue"])
        ax.plot(sub["week"], sub["fan_mean"], linewidth=lw, label=name,
                color=PALETTE["deep_blue"], linestyle=linestyle)
        if is_eliminated:
            elim_weeks = elim_by_contestant[name]
            elim_pts = sub[sub["week"].isin(elim_weeks)]
            ax.scatter(elim_pts["week"], elim_pts["fan_mean"], marker="x",
                       s=60, color=PALETTE["accent"], linewidths=1.6, zorder=4)

    apply_paper_style(ax, grid_alpha=0.2)
    ax.set_xlabel("Week")
    ax.set_ylabel("Fan Support Share")
    ax.set_title(f"Eliminations do not always align with minimum fan support (Season {season})")
    ax.scatter([], [], marker="x", s=60, color=PALETTE["accent"], label="Eliminated week")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_heatmap(accept_df: pd.DataFrame, fig_path: Path):
    if accept_df.empty:
        return
    pivot = accept_df.pivot(index="season", columns="week", values="accept_rate").fillna(0.0)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    cmap = _blue_gray_cmap()
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, origin="lower")
    cbar = fig.colorbar(im, ax=ax, label="Accept Rate (Feasible Mass Proxy)")
    cbar.ax.tick_params(colors=PALETTE["text_gray"])
    cbar.set_label("Accept Rate (Feasible Mass Proxy)", color=PALETTE["text_gray"])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_xlabel("Week")
    ax.set_ylabel("Season")
    ax.set_title("Weak identification concentrates in a few weeks (circled)")

    vals = pivot.values.copy()
    mask = np.isfinite(vals) & (vals > 0)
    if mask.any():
        flat = vals.copy()
        flat[~mask] = np.inf
        idxs = np.argsort(flat, axis=None)[:3]
        for idx in idxs:
            r, c = np.unravel_index(idx, vals.shape)
            ax.scatter(c, r, s=120, facecolors="none", edgecolors="white",
                       linewidths=1.4, alpha=0.9, zorder=3)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def compute_ppc(
    manager: ActiveSetManager,
    samples_by_week: dict,
    save_mode: JudgeSaveMode = JudgeSaveMode.J1,
    beta: float = 4.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rule_engine = RuleEngine(seed=42)
    records = []
    tension_records = []

    for (season, week), samples in samples_by_week.items():
        ctx = manager.get_season_context(int(season))
        week_ctx = ctx.weeks.get(int(week))
        if week_ctx is None or not week_ctx.has_valid_elimination():
            continue

        counts = {c: 0 for c in week_ctx.active_set}
        for s in samples:
            if ctx.voting_method == "percent":
                elim = rule_engine.percent_rule(s, week_ctx.judge_percentages, alpha=0.5).eliminated
            else:
                r_fan = rank_from_share(s)
                if ctx.has_judges_save:
                    elim = rule_engine.rank_with_judge_save(
                        r_fan,
                        week_ctx.judge_ranks,
                        judge_scores=week_ctx.judge_scores,
                        save_mode=save_mode,
                        beta=beta,
                    ).eliminated
                else:
                    elim = rule_engine.rank_rule(r_fan, week_ctx.judge_ranks).eliminated
            counts[elim] = counts.get(elim, 0) + 1

        total = max(len(samples), 1)
        probs = {c: counts.get(c, 0) / total for c in counts}
        eliminated = set(week_ctx.eliminated)

        brier = np.mean([(probs.get(c, 0) - (1 if c in eliminated else 0)) ** 2 for c in counts])
        top3 = [k for k, _ in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]]
        top3_cover = 1.0 if any(e in top3 for e in eliminated) else 0.0
        observed_prob = max([probs.get(e, 0.0) for e in eliminated]) if eliminated else 0.0
        tension_flag = 1 if observed_prob < 0.05 else 0
        tension_records.append({
            "season": season,
            "week": week,
            "observed_prob": observed_prob,
            "assumption_data_tension": tension_flag,
        })

        for c, p in probs.items():
            records.append({
                "season": season,
                "week": week,
                "contestant": c,
                "p_elim": p,
                "brier": brier,
                "top3_cover": top3_cover,
                "observed": 1 if c in eliminated else 0,
            })

    df = pd.DataFrame(records)
    summary = df.groupby("season").agg(
        top3_coverage=("top3_cover", "mean"),
        brier=("brier", "mean"),
    ).reset_index()

    df.to_csv(OUTPUT_DIR / "ppc_weekly.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "ppc_summary.csv", index=False)
    pd.DataFrame(tension_records).to_csv(OUTPUT_DIR / "assumption_data_tension_weeks.csv", index=False)
    return df, summary


def plot_ppc_metrics(summary: pd.DataFrame, fig_path: Path):
    if summary.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2))
    ax[0].bar(summary["season"], summary["top3_coverage"], color=PALETTE["deep_blue"], edgecolor=PALETTE["navy"], linewidth=0.4)
    ax[0].set_title("Top-3 Coverage")
    ax[0].set_xlabel("Season")
    ax[0].set_ylabel("Coverage")
    ax[0].grid(True, axis="y", alpha=0.2)

    ax[1].plot(summary["season"], summary["brier"], marker="o", color=PALETTE["cyan_blue"])
    ax[1].set_title("Brier Score")
    ax[1].set_xlabel("Season")
    ax[1].set_ylabel("Brier")
    ax[1].grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def plot_risk_bar(ppc_df: pd.DataFrame, fig_path: Path):
    if ppc_df.empty:
        return
    season = ppc_df["season"].max()
    subset = ppc_df[ppc_df["season"] == season]
    risk = subset.groupby("contestant")["p_elim"].mean().sort_values(ascending=False).head(12)
    plt.figure(figsize=(8.2, 4.4))
    plt.barh(risk.index[::-1], risk.values[::-1], color=PALETTE["deep_blue"], edgecolor=PALETTE["navy"], linewidth=0.4)
    plt.xlabel("Elimination Risk (avg)")
    plt.title(f"Risk Bar (Season {season})")
    plt.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def plot_sankey_audit(
    manager: ActiveSetManager,
    posterior_df: pd.DataFrame,
    fig_path: Path,
):
    """
    审计流程图（紧凑高信息密度版本）
    使用堆叠条形图替代Sankey，更紧凑美观
    """
    if posterior_df.empty:
        return

    fan_lookup = posterior_df.set_index(["season", "week", "contestant"])["fan_mean"].to_dict()
    buckets = {
        "JH_FH": 0,
        "JH_FL": 0,
        "JL_FH": 0,
        "JL_FL": 0,
    }
    outcomes = {k: {"Safe": 0, "Saved": 0, "Eliminated": 0} for k in buckets}

    for season in manager.get_all_seasons():
        ctx = manager.get_season_context(season)
        for week, week_ctx in ctx.weeks.items():
            if not week_ctx.has_valid_elimination():
                continue
            active = list(week_ctx.active_set)
            if not active:
                continue
            judge_vals = [week_ctx.judge_percentages.get(c, np.nan) for c in active]
            fan_vals = [fan_lookup.get((season, week, c), np.nan) for c in active]
            if np.all(np.isnan(judge_vals)) or np.all(np.isnan(fan_vals)):
                continue
            med_j = float(np.nanmedian(judge_vals))
            med_f = float(np.nanmedian(fan_vals))

            bottom_two = set()
            if ctx.has_judges_save:
                if ctx.voting_method == "percent":
                    combined = {}
                    for c in active:
                        j = week_ctx.judge_percentages.get(c, np.nan)
                        f = fan_lookup.get((season, week, c), np.nan)
                        if np.isnan(j) or np.isnan(f):
                            continue
                        combined[c] = 0.5 * j + 0.5 * f
                    bottom_two = {c for c, _ in sorted(combined.items(), key=lambda x: x[1])[:2]}
                else:
                    fan_shares = {c: fan_lookup.get((season, week, c), np.nan) for c in active}
                    fan_shares = {c: v for c, v in fan_shares.items() if not np.isnan(v)}
                    if len(fan_shares) >= 2:
                        fan_ranks = rank_from_share(fan_shares)
                        combined = {
                            c: fan_ranks.get(c, len(active)) + week_ctx.judge_ranks.get(c, len(active))
                            for c in active
                        }
                        bottom_two = {c for c, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:2]}

            eliminated = set(week_ctx.eliminated)
            for c in active:
                j = week_ctx.judge_percentages.get(c, np.nan)
                f = fan_lookup.get((season, week, c), np.nan)
                if np.isnan(j) or np.isnan(f):
                    continue
                bucket = ("JH_" if j >= med_j else "JL_") + ("FH" if f >= med_f else "FL")
                if bucket not in buckets:
                    continue
                buckets[bucket] += 1
                if c in eliminated:
                    outcomes[bucket]["Eliminated"] += 1
                elif c in bottom_two:
                    outcomes[bucket]["Saved"] += 1
                else:
                    outcomes[bucket]["Safe"] += 1

    total = sum(buckets.values())
    if total == 0:
        return

    # === 紧凑堆叠水平条形图 ===
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    
    categories = ["J↑ F↑", "J↑ F↓", "J↓ F↑", "J↓ F↓"]
    keys = ["JH_FH", "JH_FL", "JL_FH", "JL_FL"]
    
    safe_vals = [outcomes[k]["Safe"] for k in keys]
    saved_vals = [outcomes[k]["Saved"] for k in keys]
    elim_vals = [outcomes[k]["Eliminated"] for k in keys]
    
    y_pos = np.arange(len(categories))
    bar_height = 0.6
    
    # 堆叠条形图（低色彩 + 线型区分）
    bars1 = ax.barh(y_pos, safe_vals, bar_height, label='Safe', 
                    color=PALETTE["light_blue"], edgecolor=PALETTE["navy"], linewidth=1.2)
    bars2 = ax.barh(y_pos, saved_vals, bar_height, left=safe_vals, label='Saved',
                    color=PALETTE["light_blue"], edgecolor=PALETTE["deep_blue"], linewidth=1.2, hatch='//')
    bars3 = ax.barh(y_pos, elim_vals, bar_height, 
                    left=[s+v for s,v in zip(safe_vals, saved_vals)], label='Eliminated',
                    color=PALETTE["light_blue"], edgecolor=PALETTE["deep_blue"], linewidth=1.6, hatch='xx')
    
    # 添加数值标签（为窄段落做错位）
    min_width = 28
    for i, (s, sv, e) in enumerate(zip(safe_vals, saved_vals, elim_vals)):
        total_i = s + sv + e
        if s > 0:
            s_offset = 0.18 if s < min_width else 0.0
            ax.text(s/2, i + s_offset, f'{s}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=PALETTE["navy"])
        if sv > 0:
            sv_offset = -0.18 if sv < min_width else 0.0
            ax.text(s + sv/2, i + sv_offset, f'{sv}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=PALETTE["navy"])
        if e > 0:
            e_offset = 0.28 if e < min_width else 0.0
            ax.text(s + sv + e/2, i + e_offset, f'{e}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=PALETTE["navy"])
        # 总数标签
        ax.text(total_i + 22, i, f'n={total_i}', ha='left', va='center', fontsize=10, color=PALETTE["navy"])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Contestants', fontsize=12, fontweight='bold')
    ax.set_title('Outcomes differ systematically by judge–fan signal alignment', fontsize=13, fontweight='bold', pad=10)
    
    # 样式
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_xlim(0, max(s+sv+e for s,sv,e in zip(safe_vals, saved_vals, elim_vals)) * 1.25)

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_ridgeline_posterior(
    samples_by_week: dict,
    posterior_df: pd.DataFrame,
    fig_path: Path,
):
    if posterior_df.empty or not samples_by_week:
        return
    try:
        from scipy.stats import gaussian_kde
    except Exception:
        return

    candidates = posterior_df["contestant"].value_counts()
    target_name = None
    target_season = None
    if "Bobby Bones" in candidates.index:
        target_name = "Bobby Bones"
        seasons = posterior_df[posterior_df["contestant"] == target_name]["season"].value_counts()
        target_season = int(seasons.index[0])
    else:
        season = posterior_df.groupby("season")["week"].nunique().sort_values(ascending=False).index[0]
        df = posterior_df[posterior_df["season"] == season]
        target_name = df.groupby("contestant")["fan_mean"].mean().sort_values(ascending=False).index[0]
        target_season = int(season)

    week_samples = []
    for (season, week), samples in sorted(samples_by_week.items()):
        if int(season) != target_season:
            continue
        vals = [s.get(target_name) for s in samples if target_name in s]
        vals = np.array([v for v in vals if v is not None], dtype=float)
        if len(vals) >= 10:
            week_samples.append((int(week), vals))

    if len(week_samples) < 3:
        return

    week_samples.sort(key=lambda x: x[0])
    xs = np.linspace(0, 0.6, 200)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    offsets = np.arange(len(week_samples))
    for i, (week, vals) in enumerate(week_samples):
        kde = gaussian_kde(vals)
        dens = kde(xs)
        dens = dens / max(dens.max(), 1e-8)
        y = dens + offsets[i]
        ax.fill_between(xs, offsets[i], y, color=PALETTE["light_blue"], alpha=0.65)
        ax.plot(xs, y, color=PALETTE["cyan_blue"], linewidth=1.4)
        ax.text(0.61, offsets[i] + 0.05, f"W{week}", fontsize=8, color=PALETTE["navy"])

    ax.set_yticks([])
    ax.set_xlabel("Fan Support Share")
    ax.set_title(f"Ridgeline Posterior: {target_name} (Season {target_season})")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def plot_ternary_tradeoff(matrix_df: pd.DataFrame, pareto_df: pd.DataFrame, fig_path: Path):
    """
    绘制三元权衡图（高信息密度版本）
    增强：网格线、多策略着色、legend、区域标注
    """
    if matrix_df.empty or pareto_df.empty:
        return
    
    def _to_xy(triple):
        a, b, c = triple
        s = a + b + c
        if s <= 0:
            return (0.0, 0.0)
        a, b, c = a / s, b / s, c / s
        x = 0.5 * (2 * b + c)
        y = (np.sqrt(3) / 2) * c
        return x, y

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    
    # === 三角形边框 ===
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color=PALETTE["navy"], linewidth=2.0)
    
    # === 网格线（增加信息密度）===
    grid_levels = [0.2, 0.4, 0.6, 0.8]
    for level in grid_levels:
        # 平行于底边（fairness等值线）
        p1 = _to_xy([1-level, level/2, level/2])
        p2 = _to_xy([1-level, level, 0])
        p3 = _to_xy([1-level, 0, level])
        ax.plot([p2[0], p3[0]], [p2[1], p3[1]], color=PALETTE["ref_gray"], linewidth=0.6, linestyle="--", alpha=0.6)
    
    # === 区域着色（平衡区域高亮）===
    balance_center = _to_xy([0.33, 0.33, 0.33])
    circle = plt.Circle(balance_center, 0.15, color=PALETTE["ref_gray"], alpha=0.35, label="Balanced Zone")
    ax.add_patch(circle)
    
    # === 角标签（增大字体）===
    ax.text(-0.06, -0.06, "Fairness", color=PALETTE["navy"], fontsize=12, fontweight='bold', ha="left")
    ax.text(1.06, -0.06, "Agency", color=PALETTE["navy"], fontsize=12, fontweight='bold', ha="right")
    ax.text(0.5, np.sqrt(3) / 2 + 0.06, "Stability", color=PALETTE["navy"], fontsize=12, fontweight='bold', ha="center")
    
    # === 策略着色映射 ===
    color_map = {
        "percent": PALETTE["ref_gray"],
        "rank": PALETTE["ref_gray"],
        "rank_save": PALETTE["ref_gray"],
    }
    marker_map = {
        "percent": "o",
        "rank": "s",
        "rank_save": "^",
    }
    label_map = {
        "percent": "Percent Method",
        "rank": "Rank Method", 
        "rank_save": "Rank + Judge Save",
    }
    
    # === 绘制各策略点 ===
    label_offsets = {
        "percent": (-26, -10),
        "rank": (12, -12),
        "rank_save": (-20, 14),
    }
    label_align = {
        "percent": ("right", "top"),
        "rank": ("left", "top"),
        "rank_save": ("right", "bottom"),
    }
    plotted_labels = set()
    for row in matrix_df.itertuples(index=False):
        point = _to_xy([row.skill_alignment, row.viewer_agency, row.stability])
        system = row.system
        color = color_map.get(system, PALETTE["ref_gray"])
        marker = marker_map.get(system, "o")
        label = label_map.get(system, system) if system not in plotted_labels else None
        
        ax.scatter(point[0], point[1], s=100, facecolor="none", edgecolor=color, marker=marker,
                   linewidth=1.6, zorder=3, label=label)
        plotted_labels.add(system)
        
        # 策略标签
        dx, dy = label_offsets.get(system, (8, 8))
        ha, va = label_align.get(system, ("left", "bottom"))
        ax.annotate(system.replace("_", "+"),
                    xy=point, xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, fontweight='bold', ha=ha, va=va,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.9))

    # === DAWS 推荐点（高亮）===
    pareto_df = pareto_df.copy()
    pareto_df["stability"] = 1 - pareto_df["wrongful_rate"]
    pareto_df["min_score"] = pareto_df[["judge_alignment", "fan_influence", "stability"]].min(axis=1)
    best = pareto_df.sort_values("min_score", ascending=False).iloc[0]
    daws_point = _to_xy([best.judge_alignment, best.fan_influence, best.stability])
    
    ax.scatter(daws_point[0], daws_point[1], s=260, color=PALETTE["accent"],
               edgecolor=PALETTE["navy"], linewidth=2.4, zorder=5,
               marker="*", label="DAWS (Recommended)")
    ax.annotate("DAWS\n(α={:.2f})".format(best.get("alpha", 0.6)),
                xy=daws_point, xytext=(14, 14), textcoords="offset points",
                fontsize=10, fontweight='bold', color=PALETTE["navy"],
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["daws_gold"], alpha=0.9))
    
    # === 理想点标注 ===
    ideal_point = _to_xy([1.0, 1.0, 1.0])
    ax.scatter(ideal_point[0], ideal_point[1], s=80, color="white", 
               edgecolor=PALETTE["ref_gray"], linewidth=2, marker="D", zorder=4)
    ax.annotate("Ideal", xy=ideal_point, xytext=(8, -10), textcoords="offset points",
                fontsize=9, color=PALETTE["ref_gray"], ha="left", va="top")
    
    # === 图表设置 ===
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, np.sqrt(3) / 2 + 0.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.set_title("DAWS sits in the balanced region of the design space",
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=10, 
              bbox_to_anchor=(-0.02, 1.0))
    
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()


def run_q2_counterfactual(
    manager: ActiveSetManager,
    samples_by_week: dict,
    posterior_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rule_engine = RuleEngine(seed=7)
    systems = ["percent", "rank", "rank_save"]

    metrics = {s: {"skill": [], "agency": [], "stability": []} for s in systems}
    deficit_records = []

    for (season, week), samples in samples_by_week.items():
        ctx = manager.get_season_context(int(season))
        week_ctx = ctx.weeks.get(int(week))
        if week_ctx is None or not week_ctx.has_valid_elimination():
            continue

        elim_counts = {s: {} for s in systems}
        for s in samples:
            fan_elim = min(s.items(), key=lambda x: x[1])[0]
            e_percent = rule_engine.percent_rule(s, week_ctx.judge_percentages, alpha=0.5).eliminated
            r_fan = rank_from_share(s)
            e_rank = rule_engine.rank_rule(r_fan, week_ctx.judge_ranks).eliminated
            e_save = rule_engine.rank_with_judge_save(
                r_fan,
                week_ctx.judge_ranks,
                judge_scores=week_ctx.judge_scores,
                save_mode=JudgeSaveMode.J1,
                beta=4.0,
            ).eliminated

            for sys_name, elim in [("percent", e_percent), ("rank", e_rank), ("rank_save", e_save)]:
                elim_counts[sys_name][elim] = elim_counts[sys_name].get(elim, 0) + 1

            metrics["percent"]["agency"].append(1.0 if e_percent == fan_elim else 0.0)
            metrics["rank"]["agency"].append(1.0 if e_rank == fan_elim else 0.0)
            metrics["rank_save"]["agency"].append(1.0 if e_save == fan_elim else 0.0)

            deficit_records.append({
                "season": season,
                "week": week,
                "deficit": 1.0 if e_rank != e_percent else 0.0,
            })

        sub = posterior_df[(posterior_df["season"] == season) & (posterior_df["week"] == week)]
        if not sub.empty:
            fan_mean = dict(zip(sub["contestant"], sub["fan_mean"]))
            percent_scores = {k: 0.5 * week_ctx.judge_percentages.get(k, 0.0) + 0.5 * fan_mean.get(k, 0.0) for k in fan_mean}
            rank_scores = {k: -(rank_from_share(fan_mean).get(k, 0) + week_ctx.judge_ranks.get(k, 0)) for k in fan_mean}
            save_scores = rank_scores

            def _kendall(a: Dict[str, float], b: Dict[str, float]) -> float:
                try:
                    from scipy.stats import kendalltau
                    common = list(set(a.keys()) & set(b.keys()))
                    if len(common) < 2:
                        return float("nan")
                    av = [a[k] for k in common]
                    bv = [b[k] for k in common]
                    tau, _ = kendalltau(av, bv)
                    return float(tau) if tau is not None else float("nan")
                except Exception:
                    return float("nan")

            metrics["percent"]["skill"].append(_kendall(percent_scores, week_ctx.judge_percentages))
            metrics["rank"]["skill"].append(_kendall(rank_scores, week_ctx.judge_percentages))
            metrics["rank_save"]["skill"].append(_kendall(save_scores, week_ctx.judge_percentages))

        for sys_name in systems:
            total = sum(elim_counts[sys_name].values())
            probs = np.array([v / total for v in elim_counts[sys_name].values()]) if total > 0 else np.array([1.0])
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            norm = np.log(max(len(probs), 2))
            stability = 1.0 - entropy / norm
            metrics[sys_name]["stability"].append(stability)

    matrix_records = []
    for sys_name in systems:
        matrix_records.append({
            "system": sys_name,
            "skill_alignment": float(np.nanmean(metrics[sys_name]["skill"])) if metrics[sys_name]["skill"] else np.nan,
            "viewer_agency": float(np.mean(metrics[sys_name]["agency"])) if metrics[sys_name]["agency"] else np.nan,
            "stability": float(np.mean(metrics[sys_name]["stability"])) if metrics[sys_name]["stability"] else np.nan,
        })

    matrix_df = pd.DataFrame(matrix_records)
    deficit_df = pd.DataFrame(deficit_records)
    deficit_season = deficit_df.groupby("season")["deficit"].mean().reset_index().rename(columns={"deficit": "democratic_deficit"})

    betas = [0.5, 1.0, 2.0, 4.0, 8.0]
    sensitivity = []
    for beta in betas:
        deficits = []
        for (season, week), samples in samples_by_week.items():
            ctx = manager.get_season_context(int(season))
            week_ctx = ctx.weeks.get(int(week))
            if week_ctx is None or not week_ctx.has_valid_elimination():
                continue
            for s in samples:
                e_percent = rule_engine.percent_rule(s, week_ctx.judge_percentages, alpha=0.5).eliminated
                r_fan = rank_from_share(s)
                e_save = rule_engine.rank_with_judge_save(
                    r_fan,
                    week_ctx.judge_ranks,
                    judge_scores=week_ctx.judge_scores,
                    save_mode=JudgeSaveMode.J1,
                    beta=beta,
                ).eliminated
                deficits.append(1.0 if e_save != e_percent else 0.0)
        sensitivity.append({"beta": beta, "deficit": float(np.mean(deficits)) if deficits else np.nan})

    sensitivity_df = pd.DataFrame(sensitivity)

    matrix_df.to_csv(OUTPUT_DIR / "counterfactual_matrix.csv", index=False)
    deficit_season.to_csv(OUTPUT_DIR / "democratic_deficit.csv", index=False)
    sensitivity_df.to_csv(OUTPUT_DIR / "save_sensitivity.csv", index=False)

    return matrix_df, deficit_season, sensitivity_df


def plot_counterfactual_matrix(matrix_df: pd.DataFrame, fig_path: Path):
    if matrix_df.empty:
        return
    metrics = ["skill_alignment", "viewer_agency", "stability"]
    data = matrix_df.set_index("system")[metrics]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    cmap = _blue_gray_cmap()
    im = ax.imshow(data.values, aspect="auto", cmap=cmap)
    display_labels = []
    for name in data.index:
        if name == "percent":
            display_labels.append("Percent (DAWS base)")
        elif name == "rank_save":
            display_labels.append("Rank + Save")
        else:
            display_labels.append(name.capitalize())
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(display_labels)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    if "percent" in data.index:
        highlight_idx = data.index.get_loc("percent")
    else:
        best_label = data.min(axis=1).idxmax()
        highlight_idx = data.index.get_loc(best_label)
    ax.add_patch(plt.Rectangle((-0.5, highlight_idx - 0.5), len(metrics), 1,
                               facecolor=PALETTE["ref_gray"], alpha=0.18, edgecolor="none", zorder=2))
    cbar = fig.colorbar(im, ax=ax, label="Score")
    cbar.ax.tick_params(colors=PALETTE["text_gray"])
    cbar.set_label("Score", color=PALETTE["text_gray"])
    ax.set_title("Weighted-percent is closest to balance across metrics")
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def plot_save_sensitivity(sensitivity_df: pd.DataFrame, fig_path: Path):
    if sensitivity_df.empty:
        return
    from dwts_model.paper_palette import FIGURE_STANDARDS, apply_paper_style
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.axvspan(3.7, 4.3, color=PALETTE["light_blue"], alpha=0.15, zorder=0)
    ax.plot(sensitivity_df["beta"], sensitivity_df["deficit"], marker="o",
            color=PALETTE["deep_blue"], linewidth=FIGURE_STANDARDS["linewidth_main"])
    ax.axvline(4.0, color=PALETTE["ref_gray"], linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(4.05, sensitivity_df["deficit"].max() * 0.95, "S28+ reference",
            fontsize=9, color=PALETTE["text_gray"])
    apply_paper_style(ax, grid_alpha=0.2)
    ax.set_xlabel("Save Softness (beta)")
    ax.set_ylabel("Democratic Deficit")
    ax.set_title("Democratic deficit shifts with judge-save softness")
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def plot_democratic_deficit(deficit_df: pd.DataFrame, fig_path: Path):
    if deficit_df.empty:
        return
    from dwts_model.paper_palette import FIGURE_STANDARDS, apply_paper_style
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    x = deficit_df["season"].to_numpy()
    y = deficit_df["democratic_deficit"].to_numpy()
    if len(x) > 0:
        ax.axvspan(27.5, x.max() + 0.5, color=PALETTE["light_blue"], alpha=0.15, zorder=0)
        ax.axvline(28, color=PALETTE["ref_gray"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(0, color=PALETTE["ref_gray"], linestyle="--", linewidth=1.0, alpha=0.7)
    for xi, yi in zip(x, y):
        ax.plot([xi, xi], [0, yi], color=PALETTE["light_blue"], linewidth=1.2, alpha=0.8)
    ax.scatter(x, y, color=PALETTE["deep_blue"], edgecolor=PALETTE["navy"], s=36, zorder=3)
    apply_paper_style(ax, grid_alpha=0.2)
    ax.set_xlabel("Season")
    ax.set_ylabel("Democratic Deficit")
    ax.set_title("Rule-era shift frames the democratic deficit trend")
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def _alt_posterior_from_samples(samples_by_week: dict, sigma_alt: float = 0.25) -> pd.DataFrame:
    records = []
    prev_mean: Optional[Dict[str, float]] = None
    keys = sorted(samples_by_week.keys(), key=lambda x: (x[0], x[1]))
    for season, week in keys:
        samples = samples_by_week.get((season, week), [])
        if not samples:
            continue
        names = list(samples[0].keys())
        arr = np.array([[s.get(n, 0.0) for n in names] for s in samples], dtype=float)
        if prev_mean is None:
            mean = np.mean(arr, axis=0)
        else:
            prev = np.array([prev_mean.get(n, 0.0) for n in names], dtype=float)
            dist2 = np.sum((arr - prev) ** 2, axis=1)
            weights = np.exp(-0.5 * dist2 / max(sigma_alt ** 2, 1e-8))
            if weights.sum() <= 0:
                mean = np.mean(arr, axis=0)
            else:
                mean = np.average(arr, axis=0, weights=weights)

        prev_mean = {names[i]: float(mean[i]) for i in range(len(names))}
        for i, name in enumerate(names):
            records.append({
                "season": season,
                "week": week,
                "contestant": name,
                "fan_mean": float(mean[i]),
                "fan_hdi_low": np.nan,
                "fan_hdi_high": np.nan,
            })
    return pd.DataFrame(records)


def sync_paper_assets():
    ai_keep = set()
    generated = {
        "fig_q1_hdi_band.pdf",
        "fig_q1_uncertainty_heatmap.pdf",
        "fig_q1_risk_bar.pdf",
        "fig_q1_ppc_metrics.pdf",
        "fig_sankey_audit.pdf",
        "fig_ridgeline_posterior.pdf",
        "fig_q2_counterfactual_matrix.pdf",
        "fig_q2_save_sensitivity.pdf",
        "fig_q2_democratic_deficit.pdf",
        "fig_q3_forward_chaining.pdf",
        "fig_shap_summary.pdf",
        "fig_shap_interaction.pdf",
        "fig_shap_waterfall.pdf",
        "fig_q4_pareto_frontier.pdf",
        "fig_q4_alpha_schedule.pdf",
        "fig_q4_noise_robustness.pdf",
        "fig_ternary_tradeoff.pdf",
        "fig_dwts_show_process.pdf",
        "fig_dwts_flowchart_vector.pdf",
    }

    paper_targets = [
        PROJECT_ROOT / "paper" / "en" / "PaperC",
        PROJECT_ROOT / "paper" / "zh" / "PaperC - Chinese",
    ]

    for paper_dir in paper_targets:
        fig_dir = paper_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        for file in fig_dir.iterdir():
            if file.name in ai_keep:
                continue
            if file.name in generated:
                continue
            if file.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg", ".eps"}:
                file.unlink()

        for name in generated:
            src = FIGURES_DIR / name
            if src.exists():
                shutil.copy2(src, fig_dir / name)

        for summary_name in [
            "fan_vote_posterior_summary.csv",
            "counterfactual_matrix.csv",
            "save_sensitivity.csv",
            "democratic_deficit.csv",
            "pareto_frontier.csv",
            "ppc_summary.csv",
            "assumption_data_tension_weeks.csv",
            "accept_rate.csv",
            "gap_probability.csv",
            "forward_chaining_metrics.csv",
            "shap_sensitivity.csv",
        ]:
            src = OUTPUT_DIR / summary_name
            if src.exists():
                shutil.copy2(src, paper_dir / "sections" / summary_name)


def compile_paper(paper_dir: Path, main_file: str):
    cmd = ["latexmk", "-xelatex", "-interaction=nonstopmode", "-file-line-error", main_file]
    try:
        _run_cmd(cmd, paper_dir)
    except Exception as exc:
        print(f"[WARN] latexmk failed: {exc}")
        _run_cmd(["xelatex", "-interaction=nonstopmode", main_file], paper_dir)
        _run_cmd(["xelatex", "-interaction=nonstopmode", main_file], paper_dir)


def main():
    parser = argparse.ArgumentParser(description="Run full DWTS framework pipeline")
    parser.add_argument("--samples", type=int, default=1500, help="MaxEnt samples per week")
    parser.add_argument("--smooth-sigma", type=float, default=0.15, help="RW prior sigma")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML/XAI")
    parser.add_argument("--skip-compile", action="store_true", help="Skip LaTeX build")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loader = DWTSDataLoader(str(DATA_PATH))
    loader.load()
    manager = ActiveSetManager(loader)
    manager.build_all_contexts()

    posterior_df, intervals_df, samples_by_week, accept_df, gap_df = run_q1_inversion_sampling(
        manager, n_samples=args.samples, smooth_sigma=args.smooth_sigma
    )
    plot_hdi_band(manager, posterior_df, FIGURES_DIR / "fig_q1_hdi_band.pdf")
    plot_uncertainty_heatmap(accept_df, FIGURES_DIR / "fig_q1_uncertainty_heatmap.pdf")

    ppc_df, ppc_summary = compute_ppc(manager, samples_by_week)
    plot_ppc_metrics(ppc_summary, FIGURES_DIR / "fig_q1_ppc_metrics.pdf")
    plot_risk_bar(ppc_df, FIGURES_DIR / "fig_q1_risk_bar.pdf")
    plot_sankey_audit(manager, posterior_df, FIGURES_DIR / "fig_sankey_audit.pdf")
    plot_ridgeline_posterior(samples_by_week, posterior_df, FIGURES_DIR / "fig_ridgeline_posterior.pdf")

    matrix_df, deficit_df, sensitivity_df = run_q2_counterfactual(manager, samples_by_week, posterior_df)
    plot_counterfactual_matrix(matrix_df, FIGURES_DIR / "fig_q2_counterfactual_matrix.pdf")
    plot_save_sensitivity(sensitivity_df, FIGURES_DIR / "fig_q2_save_sensitivity.pdf")
    plot_democratic_deficit(deficit_df, FIGURES_DIR / "fig_q2_democratic_deficit.pdf")

    alphas = [round(a, 2) for a in np.linspace(0.3, 0.8, 11)]
    pareto = run_pareto_frontier(
        manager,
        posterior_df,
        alphas=alphas,
        fig_path=str(FIGURES_DIR / "fig_q4_pareto_frontier.pdf"),
    )
    pareto.结果表.to_csv(OUTPUT_DIR / "pareto_frontier.csv", index=False)
    plot_ternary_tradeoff(matrix_df, pareto.结果表, FIGURES_DIR / "fig_ternary_tradeoff.pdf")

    if not accept_df.empty:
        uncertainty = accept_df.groupby("week")["accept_rate"].mean().sort_index()
        uncertainty_list = [1 - v for v in uncertainty.values]
    else:
        uncertainty_list = []

    total_weeks = max(manager.get_season_context(s).num_weeks for s in manager.get_all_seasons())
    if len(uncertainty_list) < total_weeks:
        uncertainty_list += [0.0] * (total_weeks - len(uncertainty_list))

    alpha_schedule = compute_alpha_schedule(
        total_weeks=total_weeks,
        uncertainty=uncertainty_list[:total_weeks],
        alpha0=0.6,
        gamma=-0.1,
        eta=0.2,
        alpha_min=0.35,
        alpha_max=0.75,
        delta=0.07,
    )

    plot_alpha_schedule(alpha_schedule, str(FIGURES_DIR / "fig_q4_alpha_schedule.pdf"))
    daws_weekly = run_daws_weekly(manager, posterior_df, alpha_schedule)
    daws_weekly.to_csv(OUTPUT_DIR / "daws_weekly.csv", index=False)

    robustness_df = run_noise_robustness(manager, posterior_df, alpha_schedule, noise_level=0.05, trials=200)
    robustness_df.to_csv(OUTPUT_DIR / "daws_noise_robustness.csv", index=False)
    plot_noise_robustness(robustness_df, str(FIGURES_DIR / "fig_q4_noise_robustness.pdf"))

    if not args.skip_ml and PANEL_PATH.exists():
        ml_result = run_ml_pipeline(loader, posterior_df, str(PANEL_PATH), str(FIGURES_DIR))
        ml_result.metrics.to_csv(OUTPUT_DIR / "forward_chaining_metrics.csv", index=False)
        if ml_result.cox_summary is not None:
            ml_result.cox_summary.to_csv(OUTPUT_DIR / "cox_summary.csv", index=False)
        if ml_result.shap_importance is not None:
            ml_result.shap_importance.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)

            alt_posterior = _alt_posterior_from_samples(samples_by_week, sigma_alt=max(args.smooth_sigma * 1.5, 0.05))
            alt_fig_dir = FIGURES_DIR / "_tmp"
            alt_result = run_ml_pipeline(loader, alt_posterior, str(PANEL_PATH), str(alt_fig_dir))
            if alt_result.shap_importance is not None:
                base = ml_result.shap_importance.set_index("feature")["importance"]
                alt = alt_result.shap_importance.set_index("feature")["importance"]
                common = base.index.intersection(alt.index)
                if len(common) > 2:
                    base_rank = base.loc[common].rank(ascending=False)
                    alt_rank = alt.loc[common].rank(ascending=False)
                    rho = base_rank.corr(alt_rank, method="spearman")
                else:
                    rho = float("nan")
                pd.DataFrame([{"spearman_rho": rho}]).to_csv(OUTPUT_DIR / "shap_sensitivity.csv", index=False)

    sync_paper_assets()

    if not args.skip_compile:
        compile_paper(PROJECT_ROOT / "paper" / "en" / "PaperC", "main.tex")
        compile_paper(PROJECT_ROOT / "paper" / "zh" / "PaperC - Chinese", "main_zh.tex")

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
