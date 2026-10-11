#!/bin/bash
# run_sweep.sh POLICY.pt TASK OUTDIR  -- flat + {up,down,side} x {5,10,15,20,25} deg, 4 at a time
P=$1; TASK=$2; OUT=$3
source ~/quadbot_sim/env.sh
mkdir -p $OUT
jobs_=("0 up")
for d in up down side; do for s in 5 10 15 20 25; do jobs_+=("$s $d"); done; done
i=0
for j in "${jobs_[@]}"; do
  set -- $j
  PYTHONUNBUFFERED=1 timeout 900 python ~/quadbot_sim/eval_sweep.py $P --task $TASK --slope $1 --dir $2 --out $OUT/s$1_$2.json > $OUT/s$1_$2.log 2>&1 &
  i=$((i+1)); if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait
grep -h "^slope" $OUT/*.log | sort -k2 -n | head -300 > $OUT/all.txt
echo SWEEP_DONE
