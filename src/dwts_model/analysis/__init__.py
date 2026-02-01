# 分析模块：反事实评估、机制设计、特征归因
from .metrics import (
    计算肯德尔相关,
    计算逆转率,
    计算孔多塞一致性,
)
from .counterfactual import 反事实评估结果, 运行反事实评估
from .mechanism_design import 机制设计结果, 运行帕累托优化
from .feature_analysis import 特征分析结果, 运行特征分析
from .daws import (
    DAWSResult,
    compute_alpha_schedule,
    run_daws_weekly,
    run_noise_robustness,
    plot_alpha_schedule,
    plot_noise_robustness,
)
from .ml_xai import MLResult, run_ml_pipeline

__all__ = [
    "计算肯德尔相关",
    "计算逆转率",
    "计算孔多塞一致性",
    "反事实评估结果",
    "运行反事实评估",
    "机制设计结果",
    "运行帕累托优化",
    "特征分析结果",
    "运行特征分析",
    "MLResult",
    "run_ml_pipeline",
    "DAWSResult",
    "compute_alpha_schedule",
    "run_daws_weekly",
    "run_noise_robustness",
    "plot_alpha_schedule",
    "plot_noise_robustness",
    "run_pareto_frontier",
]


def run_pareto_frontier(*args, **kwargs):
    return 运行帕累托优化(*args, **kwargs)
