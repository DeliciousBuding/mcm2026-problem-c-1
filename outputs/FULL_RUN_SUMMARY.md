# DWTS Problem C ? Full Sample Run Summary

Run configuration:
- samples_per_week: 8000
- smooth_sigma: 0.15
- dataset: all seasons/weeks in data/2026_MCM_Problem_C_Data.csv

## Q1 ? Inversion & Uncertainty
- Accept-rate (feasible mass) mean: 0.3834 (min 0.0112, max 0.9031)
- Interval width mean: 0.8489 (median 0.9052)
- Gap probability mean: 0.4959
- PPC Top-3 coverage mean: 0.9660
- PPC Brier mean: 0.01867

Key figures:
- fig_q1_hdi_band.pdf, fig_q1_uncertainty_heatmap.pdf, fig_q1_risk_bar.pdf, fig_q1_ppc_metrics.pdf

## Q2 ? Counterfactual & Democratic Deficit
- percent: skill_alignment=0.4620, viewer_agency=0.6623, stability=0.7257
- rank: skill_alignment=0.4530, viewer_agency=0.4805, stability=0.5774
- rank_save: skill_alignment=0.4530, viewer_agency=0.3073, stability=0.5396
- Democratic deficit mean: 0.2694 (max 0.5439)

Key figures:
- fig_q2_counterfactual_matrix.pdf, fig_q2_save_sensitivity.pdf, fig_q2_democratic_deficit.pdf

## Q3 ? ML/XAI (Forward-Chaining)
- Metrics (mean across folds): auc=0.9841, brier=0.0183

Key figures:
- fig_q3_forward_chaining.pdf, fig_shap_summary.pdf, fig_shap_interaction.pdf, fig_shap_waterfall.pdf

## Q4 ? DAWS Mechanism Design
- Best alpha (max-min fairness/agency/stability): 0.8
- Best min-score: 0.4735
- Noise flip rate mean: 0.0188

Key figures:
- fig_q4_pareto_frontier.pdf, fig_q4_alpha_schedule.pdf, fig_q4_noise_robustness.pdf, fig_ternary_tradeoff.pdf

## Notes
- MILP Rank engine may fallback to heuristic for large weeks (see warning in logs).
- English paper remains within 25 pages after latest build; Chinese paper is 21 pages.