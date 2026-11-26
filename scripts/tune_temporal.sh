#!/usr/bin/env bash
# Regularization tuning on ONE assignment (train P1, validate P2).
# Only VALIDATION RMSE is used to choose settings; the test partition is not looked at.
set -u
cd "$(dirname "$0")/.."
configs=(
  "16 0.3 0.01 0"
  "32 0.3 0.01 0"
  "16 0.3 0.01 1"
  "32 0.3 0.01 1"
  "64 0.3 0.01 1"
)
for cfg in "${configs[@]}"; do
  read -r w p wd cf <<< "$cfg"
  for c in 1H 5M; do
    echo "== $c width=$w dropout=$p wd=$wd curfull=$cf"
    python3 train_temporal.py --cadence "$c" --rotation 123 --seed 42 \
      --width "$w" --dropout "$p" --wd "$wd" --curfull "$cf" \
      --tag "_tune/w${w}_d${p}_wd${wd}_cf${cf}" 2>&1 | grep -E " val |Error|error"
  done
done
echo DONE
