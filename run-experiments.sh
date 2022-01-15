#!/bin/bash
SCRIPT=${1:-mutag.py}
MODEL=${2:-gatnet}
MAX_SEED=${3:-5}

FILE_PREFIX=$(echo "$SCRIPT" | rev | cut -d'/' -f1 | rev | cut -d'.' -f1)
for SEED in `seq 1 $MAX_SEED`
do
  for DISTANCE in `seq 0 2`
  do
    for VECTOR in `seq 5 5`
    do
      RESULT_PATH="results/${FILE_PREFIX}-$SEED-$DISTANCE-$VECTOR-$MODEL.txt"
      if [ -s "$RESULT_PATH" ]
      then
        echo "Already computed: $RESULT_PATH"
      else
        python $SCRIPT $SEED $DISTANCE $VECTOR $MODEL > $RESULT_PATH
      fi
    done
  done
done
