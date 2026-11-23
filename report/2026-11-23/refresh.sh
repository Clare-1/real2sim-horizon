#!/bin/bash
# Rebuild every table and figure in this folder from horizon/results.
# Only regenerates; it never edits a number by hand. Safe to run while batches are in flight:
# partial batches simply produce smaller tables.
set -uo pipefail

OUT="$(cd "$(dirname "$0")" && pwd)"
H="$OUT/../horizon"
PY="$H/.venv-linux/bin/python"
cd "$H" || exit 1
export PYTHONPATH=src

echo "[$(date -Is)] refresh start"

run () {  # $1 = label, rest = command
  local label="$1"; shift
  if "$@" > "/tmp/refresh_$label.log" 2>&1; then echo "  ok   $label"; else echo "  FAIL $label (see /tmp/refresh_$label.log)"; fi
}

# --- tables ---
run t1  "$PY" scripts/report_experiment1.py
run t3  "$PY" scripts/report_paper8.py results/four_causes_paper8_20261120
run t4  "$PY" scripts/paired_diffs.py results/four_causes_paper8_20261120
run t2  "$PY" scripts/report_table2_full.py
run t6  "$PY" scripts/report_llm_mixed.py
run t5  "$PY" scripts/report_experiment5.py
run t7  "$PY" scripts/report_conflict_order.py results/experiment_conflict_order_20261123

# --- figures ---
for s in make_fig_r1 make_fig_r2 make_fig1_cost make_fig2_h make_fig3_action make_fig4_trajectory make_fig7_budget; do
  run "$s" "$PY" "scripts/$s.py"
done
run group "$PY" scripts/make_group_figs.py "$OUT/figures"

# --- copy tables ---
R="$H/results"
cp "$R/experiment1_table1_3banks_20261123/table1_report.md"  "$OUT/tables/T1_recipes_x_causes.md"
cp "$R/four_causes_paper8_20261120/paper8_report.md"         "$OUT/tables/T3_selectors_288.md"
cp "$R/four_causes_paper8_20261120/paired_vs_rule.md"        "$OUT/tables/T4_paired_vs_rule.md"
cp "$R/table2_full.md"                                       "$OUT/tables/T2_full_matrix.md"
cp "$R/llm_mixed_report.md"                                  "$OUT/tables/T6_llm_vs_rule_mixed.md"
cp "$R/experiment5_report.md"                                "$OUT/tables/T5_ablation_and_llm.md"
cp "$R/policy_free_signals_paper_20261123/signals.md"        "$OUT/tables/T2b_probe_signals.md"
cp "$R/calibrator_ablation_20261112/TABLE8.md"               "$OUT/tables/T8_calibrator_ablation.md"
cp "$R/experiment_conflict_order_20261123/conflict_order.md" "$OUT/tables/T7_order_partial.md"
cp "$R/fig4_trajectory_paper8_20261120/fig4_chains.md"       "$OUT/tables/fig4_chains.md"
cp "$R/fig1_sr_cost_20261122/fig1_numbers.md"                    "$OUT/tables/fig1_numbers.md"
cp "$R/fig2_sr_vs_h_20261123/fig2_numbers.md"                    "$OUT/tables/fig2_numbers.md"
cp "$R/fig7_budget_20261123/fig7_numbers.md"                     "$OUT/tables/fig7_numbers.md"
cp "$R/fig_R1_recipe_cause_20261121/fig_R1_recipe_cause.md"      "$OUT/tables/figR1_numbers.md"
cp "$R/fig_R2_regret_20261122/fig_R2_regret.md"                  "$OUT/tables/figR2_numbers.md"

# --- copy figures ---
cp "$R/fig_R1_recipe_cause_20261121/fig_R1_recipe_cause.png"           "$OUT/figures/figR1_recipe_cause.png"
cp "$R/fig_R2_regret_20261122/fig_R2_regret.png"                       "$OUT/figures/figR2_regret.png"
cp "$R/fig1_sr_cost_20261122/fig1_sr_cost.png"                         "$OUT/figures/fig1_sr_vs_cost.png"
cp "$R/fig2_sr_vs_h_20261123/fig2_sr_vs_h.png"                         "$OUT/figures/fig2_sr_vs_h.png"
cp "$R/fig3_action_four_causes_paper8_20261120/fig3_first_action.png"  "$OUT/figures/fig3_first_action.png"
cp "$R/fig3_action_four_causes_paper8_20261120/fig3_first_action_by_selector.png" "$OUT/figures/fig3_first_action_by_selector.png"
cp "$R/fig4_trajectory_paper8_20261120/fig4_trajectory.png"            "$OUT/figures/fig4_trajectory.png"
cp "$R/paper_figures_20261116/fig7_budget_sweep.png"                   "$OUT/figures/fig7_budget_sweep.png"

# --- paper ---
cp "$OUT/../paper-template-latex/rss2027_submission.pdf" "$OUT/paper/" 2>/dev/null
cp "$OUT/../paper-template-latex/rss2027_submission.tex" "$OUT/paper/" 2>/dev/null

{
  echo ""
  echo "| $(date -Is) | order $(find "$R/experiment_conflict_order_20261123" -name record.json 2>/dev/null | wc -l)/36 | a1 $(find "$R/experiment_a1_perturbed_20261123" -name record.json 2>/dev/null | wc -l)/24 | paper_llm $(find "$R/paper_llm_rows_20261123" -name record.json 2>/dev/null | wc -l)/72 |"
} >> "$OUT/REFRESH_LOG.md"

echo "[$(date -Is)] refresh done"