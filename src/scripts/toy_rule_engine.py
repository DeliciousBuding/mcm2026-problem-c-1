from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dwts_model.engines.rule_engine import (
    RuleEngine,
    JudgeSaveMode,
    compute_judge_percentages,
    rank_from_share,
    rank_from_scores,
)


def main() -> None:
    engine = RuleEngine(seed=42)

    # Toy inputs
    v_share = {"A": 0.2, "B": 0.3, "C": 0.5}
    judge_scores = {"A": 20.0, "B": 30.0, "C": 10.0}
    j_pct = compute_judge_percentages(judge_scores)

    # Percent rule: lowest combined should be eliminated (A)
    percent = engine.percent_rule(v_share, j_pct, alpha=0.5)
    assert percent.eliminated == "A", f"Percent rule failed: {percent}"

    # Rank rule: worst combined rank should be eliminated (A)
    r_fan = rank_from_share(v_share)          # C=1, B=2, A=3
    r_judge = rank_from_scores(judge_scores)  # B=1, A=2, C=3
    rank = engine.rank_rule(r_fan, r_judge)
    assert rank.eliminated == "A", f"Rank rule failed: {rank}"

    # Rank + judge save (J2): save higher judge score among bottom two
    rank_j2 = engine.rank_with_judge_save(
        r_fan,
        r_judge,
        judge_scores=judge_scores,
        save_mode=JudgeSaveMode.J2,
    )
    assert set(rank_j2.bottom_two or []) == {"A", "C"}, f"Bottom two mismatch: {rank_j2}"
    assert rank_j2.saved == "A" and rank_j2.eliminated == "C", f"J2 failed: {rank_j2}"

    # J0/J1 are stochastic; just ensure elimination comes from bottom two.
    rank_j0 = engine.rank_with_judge_save(
        r_fan,
        r_judge,
        judge_scores=judge_scores,
        save_mode=JudgeSaveMode.J0,
    )
    assert rank_j0.eliminated in (rank_j0.bottom_two or ()), f"J0 failed: {rank_j0}"

    rank_j1 = engine.rank_with_judge_save(
        r_fan,
        r_judge,
        judge_scores=judge_scores,
        save_mode=JudgeSaveMode.J1,
        beta=4.0,
    )
    assert rank_j1.eliminated in (rank_j1.bottom_two or ()), f"J1 failed: {rank_j1}"

    print("Toy rule-engine checks passed.")
    print("Percent eliminated:", percent.eliminated)
    print("Rank eliminated:", rank.eliminated)
    print("Rank+Save J2 eliminated:", rank_j2.eliminated, "saved:", rank_j2.saved)
    print("Rank+Save J0 eliminated:", rank_j0.eliminated, "saved:", rank_j0.saved)
    print("Rank+Save J1 eliminated:", rank_j1.eliminated, "saved:", rank_j1.saved)


if __name__ == "__main__":
    main()
