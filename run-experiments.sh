#!/bin/bash
MODEL="gatnet"
SCRIPT="mutag.py"
FILE_PREFIX=$(echo "$SCRIPT" | rev | cut -d'/' -f1 | rev | cut -d'.' -f1)
for SEED in `seq 1 5`
do
  for DISTANCE in `seq 0 2`
  do
    for VECTOR in `seq 5 5`
    do
      python $SCRIPT $SEED $DISTANCE $VECTOR $MODEL > results/${FILE_PREFIX}-$SEED-$DISTANCE-$VECTOR-$MODEL.txt
    done
  done
done
