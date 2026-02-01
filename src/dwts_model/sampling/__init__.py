# 采样模块：截断贝叶斯 + MCMC
from .bayes_mcmc import (
    采样_单周,
    汇总后验,
    计算_hdi,
    hit_and_run_sample,
    importance_resample_with_prior,
    project_to_simplex_with_bounds,
)


def sample_week(*args, **kwargs):
    return 采样_单周(*args, **kwargs)


def summarize_posterior(*args, **kwargs):
    return 汇总后验(*args, **kwargs)

__all__ = [
    "采样_单周",
    "汇总后验",
    "计算_hdi",
    "hit_and_run_sample",
    "importance_resample_with_prior",
    "project_to_simplex_with_bounds",
    "sample_week",
    "summarize_posterior",
]
