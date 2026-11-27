#!/usr/bin/env bash
# Everything after training (scripts/rerun_basic_and_explain.sh + scripts/run_final.sh):
# statistics, analyses for the 1-hour (5M) and 24-hour (1H) models, figures, and web exports.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/logs
python3 src/final_stats.py
for c in 1H 5M; do for r in 123 132 213 231 312 321; do
  python3 src/explain_errors.py $c $r 42 results/temporal_final > results/logs/explain_${c}_$r.log 2>&1 || echo "FAIL explain $c $r"
  python3 src/history_importance.py $c $r 42 > results/logs/history_${c}_$r.log 2>&1 || echo "FAIL history $c $r"
  python3 src/variable_importance.py $c $r 42 > results/logs/vars_${c}_$r.log 2>&1 || echo "FAIL vars $c $r"
  python3 src/occlusion.py $c $r 42 > results/logs/occl_${c}_$r.log 2>&1 || echo "FAIL occlusion $c $r"
  python3 src/occlusion_all.py $c $r 42 > results/logs/occlall_${c}_$r.log 2>&1 || echo "FAIL occlusion_all $c $r"
done; done
python3 scripts/track_training.py > results/logs/track.log 2>&1 || echo "FAIL track"
python3 src/plot_results.py results/temporal_final > /dev/null 2>&1 || echo "FAIL plot_results"
python3 src/plot_tracking.py > /dev/null 2>&1 || echo "FAIL plot_tracking"
python3 src/plot_best_vs_nairas.py > /dev/null 2>&1 || echo "FAIL plot_best"
python3 src/paper_figures.py > results/logs/paper_figures.log 2>&1 || echo "FAIL paper_figures"
python3 src/plot_explain.py > /dev/null 2>&1 || echo "FAIL plot_explain"
python3 src/export_web.py results/temporal_final > /dev/null 2>&1 && python3 scripts/export_animation.py > /dev/null && python3 poster/animation/build.py > /dev/null 2>&1 || echo "FAIL web/animation"
python3 scripts/export_model_zoo.py > results/logs/zoo.log 2>&1 && python3 poster/zoo/build.py > /dev/null || echo "FAIL zoo"
python3 scripts/export_tutorial.py > results/logs/tutorial.log 2>&1 && python3 poster/models_explained/build.py > /dev/null || echo "FAIL tutorial"
echo ANALYSIS_DONE
