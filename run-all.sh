#!/bin/bash

NUM_EXPERIMENTS=10
PIDS=""

# Run all experiments in parallel but each model type and config sequentially
for SCRIPT_FILE in "Zinc12k" "mutag" "enzymes" "enzymes_contfeat" "proteins" "ptc"
do
  echo "$SCRIPT_FILE is now running..."
  {
    for MODEL_TYPE in "gatnet" "gcnnet" "ginnet" "chebnet" "mlpnet" "gnnml1" "gnnml3"
    do
      echo "Running ./run-experiments.sh ${SCRIPT_FILE}.py $MODEL_TYPE $NUM_EXPERIMENTS"
      ./run-experiments.sh ${SCRIPT_FILE}.py $MODEL_TYPE $NUM_EXPERIMENTS
    done
  } &
  PIDS="$PIDS $!"
done

echo "Starting wait until the processes finish..."

wait $PIDS
