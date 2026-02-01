"""
Rule engine for percent and rank mechanisms (with judge save variants).

Notation mapping:
- v_{i,t}: fan support share (continuous, sums to 1)
- r^{fan}_{i,t}: fan rank (1 = best)
- J_{i,t}: judge score sum
- Jpct_{i,t}: judge percent (J_{i,t} / sum_k J_{k,t})
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


class RuleMode(str, Enum):
    PERCENT = "percent"
    RANK = "rank"


class JudgeSaveMode(str, Enum):
    NONE = "none"
    J0 = "J0"  # random save
    J1 = "J1"  # softmax by judge score
    J2 = "J2"  # deterministic by judge score


@dataclass
class RuleResult:
    eliminated: str
    saved: Optional[str]
    bottom_two: Optional[Tuple[str, str]]
    combined: Dict[str, float]


def compute_judge_percentages(judge_scores: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(judge_scores.values()))
    if total <= 0:
        n = max(len(judge_scores), 1)
        return {k: 1.0 / n for k in judge_scores}
    return {k: v / total for k, v in judge_scores.items()}


def rank_from_scores(
    scores: Dict[str, float],
    higher_is_better: bool = True,
) -> Dict[str, int]:
    # Rank 1 is best. Ties are broken by name to keep determinism.
    if higher_is_better:
        ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    else:
        ordered = sorted(scores.items(), key=lambda x: (x[1], x[0]))
    return {name: i + 1 for i, (name, _) in enumerate(ordered)}


def rank_from_share(v_share: Dict[str, float]) -> Dict[str, int]:
    return rank_from_scores(v_share, higher_is_better=True)


def _ensure_keys(source: Dict[str, float], keys: Iterable[str], default: float) -> Dict[str, float]:
    return {k: source.get(k, default) for k in keys}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _bottom_two_candidates(
    combined_ranks: Dict[str, float],
    r_fan: Dict[str, int],
    r_judge: Dict[str, int],
) -> Tuple[str, str]:
    # Worst two by combined rank, then by worse fan rank, then worse judge rank, then name.
    ordered = sorted(
        combined_ranks.keys(),
        key=lambda name: (
            -combined_ranks.get(name, 0.0),
            -r_fan.get(name, 0),
            -r_judge.get(name, 0),
            name,
        ),
    )
    if len(ordered) < 2:
        raise ValueError("Need at least two contestants for bottom-two selection")
    return ordered[0], ordered[1]


class RuleEngine:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def percent_rule(
        self,
        v_share: Dict[str, float],
        judge_percentages: Dict[str, float],
        alpha: float = 0.5,
    ) -> RuleResult:
        contestants = list(v_share.keys())
        j_pct = _ensure_keys(judge_percentages, contestants, 0.0)
        combined = {k: alpha * j_pct.get(k, 0.0) + (1 - alpha) * v_share.get(k, 0.0) for k in contestants}
        eliminated = min(combined.items(), key=lambda x: (x[1], x[0]))[0]
        return RuleResult(eliminated=eliminated, saved=None, bottom_two=None, combined=combined)

    def rank_rule(
        self,
        r_fan: Dict[str, int],
        r_judge: Dict[str, int],
    ) -> RuleResult:
        contestants = list(r_fan.keys())
        rj = _ensure_keys({k: float(v) for k, v in r_judge.items()}, contestants, float(len(contestants)))
        combined = {k: float(r_fan.get(k, len(contestants))) + float(rj.get(k, len(contestants))) for k in contestants}
        eliminated = max(combined.items(), key=lambda x: (x[1], x[0]))[0]
        bottom_two = _bottom_two_candidates(combined, r_fan, {k: int(v) for k, v in rj.items()})
        return RuleResult(eliminated=eliminated, saved=None, bottom_two=bottom_two, combined=combined)

    def rank_with_judge_save(
        self,
        r_fan: Dict[str, int],
        r_judge: Dict[str, int],
        judge_scores: Optional[Dict[str, float]] = None,
        save_mode: JudgeSaveMode = JudgeSaveMode.J1,
        beta: float = 4.0,
    ) -> RuleResult:
        base = self.rank_rule(r_fan, r_judge)
        if save_mode == JudgeSaveMode.NONE:
            return base

        bottom_two = base.bottom_two
        if bottom_two is None:
            return base

        a, b = bottom_two

        strength = {}
        if judge_scores and len(judge_scores) > 0:
            strength[a] = float(judge_scores.get(a, 0.0))
            strength[b] = float(judge_scores.get(b, 0.0))
        else:
            strength[a] = -float(r_judge.get(a, 0))
            strength[b] = -float(r_judge.get(b, 0))

        if save_mode == JudgeSaveMode.J0:
            saved = self.rng.choice([a, b])
        elif save_mode == JudgeSaveMode.J1:
            weights = _softmax(np.array([beta * strength[a], beta * strength[b]], dtype=float))
            saved = self.rng.choice([a, b], p=weights)
        elif save_mode == JudgeSaveMode.J2:
            if strength[a] > strength[b]:
                saved = a
            elif strength[b] > strength[a]:
                saved = b
            else:
                # Tie-breaker: better (lower) fan rank, then name.
                saved = a if (r_fan.get(a, 0), a) < (r_fan.get(b, 0), b) else b
        else:
            raise ValueError(f"Unknown save_mode: {save_mode}")

        eliminated = b if saved == a else a
        return RuleResult(eliminated=eliminated, saved=saved, bottom_two=bottom_two, combined=base.combined)
