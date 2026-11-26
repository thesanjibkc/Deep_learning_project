#!/usr/bin/env bash
# Final runs with the setting chosen on validation (rotation 123):
# width 16, dropout 0.3, weight decay 0.01. All 6 rotations x 3 seeds x both cadences.
# Extra: residual Transformer with weighted Huber loss (x4 for A > 15).
set -u
cd "$(dirname "$0")/.."
for s in 42 1 2; do
  for c in 1H 5M; do
    for r in 123 132 213 231 312 321; do
      echo "== $c rot $r seed $s"
      python3 train_temporal.py --cadence "$c" --rotation "$r" --seed "$s" \
        --width 16 --dropout 0.3 --wd 0.01 --tag _final 2>&1 | grep -E " val |rror"
      python3 train_temporal.py --cadence "$c" --rotation "$r" --seed "$s" \
        --width 16 --dropout 0.3 --wd 0.01 --models Transformer --loss huber --hiw 4 \
        --tag _final_weighted 2>&1 | grep -E " val |rror"
    done
  done
done
echo DONE
