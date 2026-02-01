"""
Forward-chaining ML + Cox + SHAP for Q3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from ..paper_palette import PALETTE


@dataclass
class MLResult:
    metrics: pd.DataFrame
    shap_summary_path: Optional[str]
    shap_interaction_path: Optional[str]
    shap_waterfall_path: Optional[str]
    cox_summary: Optional[pd.DataFrame]
    shap_importance: Optional[pd.DataFrame]


def _blue_gray_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "dwts_blue_gray",
        [PALETTE.get("pale_blue", "#E6F2F8"), PALETTE["ref_gray"], PALETTE["deep_blue"]],
        N=256,
    )


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        from sklearn.metrics import brier_score_loss
        return float(brier_score_loss(y_true, y_prob))
    except Exception:
        return float("nan")


def _safe_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        from sklearn.metrics import log_loss
        return float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        return float("nan")


def _build_pro_tier(contestants: pd.DataFrame) -> pd.DataFrame:
    stats = (
        contestants.groupby("ballroom_partner")
        .agg(avg_placement=("placement", "mean"), win_rate=("status", lambda x: np.mean(x == "winner")))
        .reset_index()
    )
    # Lower placement is better; tier 3 = top pros
    try:
        stats["pro_tier"] = pd.qcut(stats["avg_placement"], 3, labels=[1, 2, 3], duplicates="drop")
    except Exception:
        stats["pro_tier"] = np.nan
    stats["pro_tier"] = stats["pro_tier"].astype(float)
    return stats


def build_weekly_dataset(
    loader,
    posterior_df: pd.DataFrame,
    panel_path: str,
) -> pd.DataFrame:
    weekly = pd.read_csv(panel_path)
    weekly["season"] = weekly["season"].astype(int)
    weekly["week"] = weekly["week"].astype(int)

    contestants = loader.processed_df.copy()
    contestants = contestants[[
        "season",
        "celebrity_name",
        "elimination_week",
        "status",
        "placement",
        "celebrity_age_during_season",
        "celebrity_industry",
        "ballroom_partner",
    ]]

    weekly = weekly.merge(contestants, on=["season", "celebrity_name"], how="left")

    # active weeks
    weekly["elimination_week"] = pd.to_numeric(weekly["elimination_week"], errors="coerce")
    weekly["active"] = True
    weekly.loc[weekly["status"] == "withdrew", "active"] = False
    weekly.loc[weekly["elimination_week"].notna() & (weekly["week"] > weekly["elimination_week"]), "active"] = False

    weekly = weekly[weekly["active"]].copy()
    weekly["eliminated"] = (weekly["elimination_week"].notna() & (weekly["week"] == weekly["elimination_week"])).astype(int)

    # posterior fan mean
    if posterior_df is not None and not posterior_df.empty:
        fan = posterior_df.rename(columns={"contestant": "celebrity_name"})[["season", "week", "celebrity_name", "fan_mean"]]
        weekly = weekly.merge(fan, on=["season", "week", "celebrity_name"], how="left")
    else:
        weekly["fan_mean"] = np.nan

    # partner tier
    pro_stats = _build_pro_tier(contestants.dropna(subset=["ballroom_partner"]))
    weekly = weekly.merge(pro_stats, on="ballroom_partner", how="left")

    weekly["pageviews"] = pd.to_numeric(weekly.get("pageviews", np.nan), errors="coerce")

    return weekly


def forward_chaining_xgb(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "eliminated",
    group_col: str = "season",
) -> Tuple[pd.DataFrame, Optional[object]]:
    seasons = sorted(df[group_col].dropna().unique())
    records = []
    final_model = None

    try:
        import xgboost as xgb
        use_xgb = True
    except Exception:
        from sklearn.linear_model import LogisticRegression
        use_xgb = False

    for season in seasons:
        train = df[df[group_col] < season]
        test = df[df[group_col] == season]
        if len(train) < 50 or len(test) < 10:
            continue

        X_train = train[feature_cols]
        y_train = train[target_col].astype(int)
        X_test = test[feature_cols]
        y_test = test[target_col].astype(int)

        if use_xgb:
            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
            )
            model.fit(X_train, y_train)
        else:
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:, 1]
        auc = _safe_auc(y_test.to_numpy(), prob)
        brier = _safe_brier(y_test.to_numpy(), prob)
        logloss = _safe_logloss(y_test.to_numpy(), prob)

        records.append({
            "season": season,
            "auc": auc,
            "brier": brier,
            "logloss": logloss,
            "n_test": len(test),
        })
        final_model = model

    return pd.DataFrame(records), final_model


def run_cox_model(df: pd.DataFrame, feature_cols: List[str]) -> Optional[pd.DataFrame]:
    # contestant-level summary
    grouped = df.groupby(["season", "celebrity_name"], as_index=False).agg(
        duration=("week", "max"),
        event=("eliminated", "max"),
        judge_total=("judge_total", "mean"),
        fan_mean=("fan_mean", "mean"),
        age=("celebrity_age_during_season", "mean"),
        pro_tier=("pro_tier", "mean"),
    )
    grouped = grouped.dropna(subset=["duration"])

    try:
        from lifelines import CoxPHFitter
        cph = CoxPHFitter()
        cph.fit(grouped[["duration", "event"] + feature_cols], duration_col="duration", event_col="event")
        summary = cph.summary.reset_index().rename(columns={"index": "feature"})
        return summary
    except Exception:
        return None


def _plot_forward_chaining(metrics: pd.DataFrame, fig_path: str):
    """绘制前向验证图（高信息密度版本）"""
    if metrics.empty:
        return
    
    from ..paper_palette import FIGURE_STANDARDS, apply_paper_style
    
    fig, ax = plt.subplots(figsize=FIGURE_STANDARDS["figsize_standard"])
    
    seasons = metrics["season"].to_numpy()
    auc = metrics["auc"].to_numpy()
    brier_comp = 1 - metrics["brier"].to_numpy()
    
    # 双线图
    ax.plot(seasons, auc, marker="o", color=PALETTE["deep_blue"], 
            linewidth=FIGURE_STANDARDS["linewidth_main"], 
            markersize=8, markeredgecolor=PALETTE["baseline"], markeredgewidth=1.5,
            label="AUC", zorder=3)
    ax.plot(seasons, brier_comp, marker="s", color=PALETTE["cyan_blue"], 
            linewidth=FIGURE_STANDARDS["linewidth_secondary"], 
            markersize=8, markeredgecolor=PALETTE["baseline"], markeredgewidth=1.5,
            label="1 - Brier Score", zorder=3)
    
    # 填充性能区间
    ax.fill_between(seasons, auc, brier_comp, alpha=0.12, color=PALETTE["fill"])
    
    # 参考线
    ax.axhline(y=0.5, color=PALETTE["ref_gray"], linestyle="--", linewidth=1.2, 
               alpha=0.5, label="Random Baseline")
    
    apply_paper_style(ax)
    ax.set_xlabel("Season", fontsize=FIGURE_STANDARDS["label_fontsize"])
    ax.set_ylabel("Score", fontsize=FIGURE_STANDARDS["label_fontsize"])
    ax.set_title("Predictive scores stay high under forward-chaining", fontsize=FIGURE_STANDARDS["title_fontsize"], fontweight='bold')
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=FIGURE_STANDARDS["legend_fontsize"])
    ax.set_ylim(0.4, 1.02)
    ax.set_yticks(np.arange(0.4, 1.01, 0.2))
    
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_shap_summary(
    model,
    X: pd.DataFrame,
    fig_path: str,
    max_display: Optional[int] = 15,
) -> Tuple[bool, Optional[pd.DataFrame]]:
    try:
        import shap
        shap_values = shap.TreeExplainer(model).shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        cmap = _blue_gray_cmap()
        try:
            shap.summary_plot(
                shap_values,
                X,
                show=False,
                plot_size=(7, 4.8),
                cmap=cmap,
                max_display=max_display,
            )
        except TypeError:
            shap.summary_plot(
                shap_values,
                X,
                show=False,
                plot_size=(7, 4.8),
                color=cmap,
                max_display=max_display,
            )
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        importance = np.mean(np.abs(shap_values), axis=0)
        imp_df = pd.DataFrame({"feature": X.columns, "importance": importance}).sort_values("importance", ascending=False)
        return True, imp_df
    except Exception:
        # Fallback to model-based importance
        importance = None
        if hasattr(model, "feature_importances_"):
            importance = np.array(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coef = np.array(model.coef_).ravel()
            importance = np.abs(coef)

        if importance is None or len(importance) != X.shape[1]:
            return False, None

        imp_df = pd.DataFrame({"feature": X.columns, "importance": importance}).sort_values("importance", ascending=False)
        top = imp_df.head(12)
        plt.figure(figsize=(7, 4.6))
        plt.barh(top["feature"][::-1], top["importance"][::-1], color=PALETTE["deep_blue"])
        plt.xlabel("Importance (Fallback)")
        plt.title("Feature Importance (Fallback)")
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        return True, imp_df


def _plot_shap_interaction(model, X: pd.DataFrame, fig_path: str) -> bool:
    if "celebrity_age_during_season" not in X.columns or "pro_tier" not in X.columns:
        return False
    try:
        import shap
        shap_values = shap.TreeExplainer(model).shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        cmap = _blue_gray_cmap()
        try:
            shap.dependence_plot(
                "celebrity_age_during_season",
                shap_values,
                X,
                interaction_index="pro_tier",
                show=False,
                cmap=cmap,
            )
        except TypeError:
            shap.dependence_plot(
                "celebrity_age_during_season",
                shap_values,
                X,
                interaction_index="pro_tier",
                show=False,
            )
        plt.title("Age effect is nonlinear and moderated by pro tier")
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[:, 1]
            else:
                prob = model.predict(X)
            plt.figure(figsize=(6.8, 4.6))
            plt.scatter(
                X["celebrity_age_during_season"],
                X["pro_tier"],
                c=prob,
                cmap=_blue_gray_cmap(),
                s=16,
                alpha=0.7,
                edgecolors="none",
            )
            plt.xlabel("Celebrity Age")
            plt.ylabel("Pro Tier")
            plt.title("Age × Pro Tier (Fallback)")
            cbar = plt.colorbar(label="Predicted Risk")
            cbar.ax.tick_params(colors=PALETTE["text_gray"])
            cbar.set_label("Predicted Risk", color=PALETTE["text_gray"])
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches="tight")
            plt.close()
            return True
        except Exception:
            return False


def _plot_shap_waterfall(model, X: pd.DataFrame, y: Optional[pd.Series], fig_path: str) -> bool:
    try:
        import shap
    except Exception:
        return False

    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[:, 1]
        else:
            prob = model.predict(X)
    except Exception:
        return False

    idx = int(np.argmax(prob))
    if y is not None:
        mask = (y.astype(int) == 0)
        if mask.any():
            candidate = np.where(mask)[0]
            idx = int(candidate[np.argmax(prob[candidate])])

    try:
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X)
        shap.plots.waterfall(explanation[idx], show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        try:
            shap_values = shap.TreeExplainer(model).shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            values = shap_values[idx]
            imp = pd.DataFrame({"feature": X.columns, "impact": values})
            imp = imp.reindex(imp["impact"].abs().sort_values().index).tail(12)
            cmap = _blue_gray_cmap()
            colors = [
                PALETTE["ref_gray"] if abs(v) < 1e-4 else (cmap(0.2) if v < 0 else cmap(0.8))
                for v in imp["impact"]
            ]
            plt.figure(figsize=(6.4, 4.2))
            plt.barh(imp["feature"], imp["impact"], color=colors)
            plt.xlabel("SHAP Impact")
            plt.title("SHAP Waterfall (Fallback)")
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches="tight")
            plt.close()
            return True
        except Exception:
            return False


def run_ml_pipeline(
    loader,
    posterior_df: pd.DataFrame,
    panel_path: str,
    fig_dir: str,
) -> MLResult:
    from pathlib import Path
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    df = build_weekly_dataset(loader, posterior_df, panel_path)

    # feature prep
    base_features = [
        "judge_total",
        "judge_rank",
        "judge_share",
        "pageviews",
        "fan_mean",
        "celebrity_age_during_season",
        "pro_tier",
        "win_rate",
    ]
    for col in base_features:
        if col not in df.columns:
            df[col] = np.nan

    df["pageviews"] = df["pageviews"].fillna(0.0)
    df["fan_mean"] = df["fan_mean"].fillna(df["fan_mean"].median())

    # categorical: industry
    df = pd.get_dummies(df, columns=["celebrity_industry"], drop_first=True)

    feature_cols = base_features + [c for c in df.columns if c.startswith("celebrity_industry_")]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median()).fillna(0.0)

    metrics, model = forward_chaining_xgb(df, feature_cols)

    shap_summary_path = None
    shap_interaction_path = None
    shap_waterfall_path = None
    shap_importance = None
    if model is not None:
        X_all = df[feature_cols]
        shap_summary_path = f"{fig_dir}/fig_shap_summary.pdf"
        shap_interaction_path = f"{fig_dir}/fig_shap_interaction.pdf"
        _, shap_importance = _plot_shap_summary(model, X_all, shap_summary_path, max_display=15)
        # full summary for appendix
        _plot_shap_summary(model, X_all, f"{fig_dir}/fig_shap_summary_full.pdf", max_display=None)
        _plot_shap_interaction(model, X_all, shap_interaction_path)
        shap_waterfall_path = f"{fig_dir}/fig_shap_waterfall.pdf"
        _plot_shap_waterfall(model, X_all, df["eliminated"] if "eliminated" in df.columns else None, shap_waterfall_path)

    _plot_forward_chaining(metrics, f"{fig_dir}/fig_q3_forward_chaining.pdf")

    cox_summary = run_cox_model(df, ["judge_total", "fan_mean", "celebrity_age_during_season", "pro_tier"])

    return MLResult(
        metrics=metrics,
        shap_summary_path=shap_summary_path,
        shap_interaction_path=shap_interaction_path,
        shap_waterfall_path=shap_waterfall_path,
        cox_summary=cox_summary,
        shap_importance=shap_importance,
    )
