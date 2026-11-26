#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
for s in 42 1 2; do for r in 123 132 213 231 312 321; do
  python3 train_basic.py --rotation $r --seed $s > /dev/null 2>&1 || echo "FAIL basic $r $s"
done; done
echo BASIC_DONE
for r in 123 132 213 231 312 321; do
  python3 src/explain_errors.py 1H $r 42 results/temporal_final > results/logs/explain_$r.log 2>&1 || echo "FAIL explain $r"
done
echo DONE
