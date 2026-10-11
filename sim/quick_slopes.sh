#!/bin/bash
# quick_slopes.sh POLICY TASK OUTDIR -- the steep cases only (20/25 deg up + side), in parallel
P=$1; TASK=$2; OUT=$3
source ~/quadbot_sim/env.sh; mkdir -p $OUT
for j in "20 side" "25 side" "20 up" "25 up"; do set -- $j
  PYTHONUNBUFFERED=1 timeout 600 python ~/quadbot_sim/eval_sweep.py $P --task $TASK --slope $1 --dir $2 --speeds 0.2,0.4,0.6,0.8 --out $OUT/s$1_$2.json > $OUT/s$1_$2.log 2>&1 &
done; wait; grep -h "^slope" $OUT/*.log
