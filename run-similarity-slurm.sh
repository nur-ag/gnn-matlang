#!/bin/bash
#SBATCH -J GNN-Sim
#SBATCH -p high
#SBATCH -n 1 #number of tasks
#SBATCH -c 1
#SBATCH --mem=16384
#SBATCH --array=1-210:1
#SBATCH --output=results/execution_logs/run.%A_%a.out

module load PyTorch-Geometric/2.0.2-foss-2020b-PyTorch-1.10.0

# This is required to load Python path w/ local installs
# Please comment it out / delete it if reproducing, it's a hack
export PYTHONPATH="/homedtic/fnalvarez/.local/lib/python3.8/site-packages/:$PYTHONPATH"

# Output directory defaults to results
OUTPUT_DIR=${1:-results/similarity/}

# Experiment constants
MAX_SEED=10
DEVICE='cpu' # We use CPU (slower) since it lets us run more jobs in parallel

# Generate all the commands and their outputs
COMMANDS_ARRAY=()
OUTPUTS_ARRAY=()
for SCRIPT in "exp_classify" "graph8c" "sr25"
do
  for MODEL_TYPE in "gatnet" "gcnnet" "ginnet" "chebnet" "mlpnet" "gnnml3"
  do
    for SEED in `seq 1 $MAX_SEED`
    do
      for DISTANCE in `seq 0 2`
      do
        LENGTHS="0" && [[ $DISTANCE -gt 0 ]] && LENGTHS="-1 5 10"
        for VECTOR_LENGTH in $LENGTHS
        do
          CMD="python ${SCRIPT}.py $SEED $DISTANCE $VECTOR_LENGTH $MODEL_TYPE $DEVICE";
          LENGTH_OR_ENC="${VECTOR_LENGTH}" && [[ $VECTOR_LENGTH -le 0 ]] && LENGTH_OR_ENC="Encoding"
          OUTPUT="$OUTPUT_DIR/${SCRIPT}-${SEED}-${DISTANCE}-${LENGTH_OR_ENC}-${MODEL_TYPE}-${DEVICE}.txt"
          COMMANDS_ARRAY+=("$CMD")
          OUTPUTS_ARRAY+=("$OUTPUT")
        done
      done
    done
  done
done
# Total combinations/experiments:
# 2 scripts x 6 models x 10 seeds x 3 distances x 3 lengths = 1080
TOTAL_EXPERIMENTS=${#COMMANDS_ARRAY[@]}
TOTAL_PROCESSES=$SLURM_ARRAY_TASK_COUNT

# Run through all the possible processes collaboratively
# Or just run sequentially if we're not on a Slurm cluster.
if [[ ! -z "$SLURM_ARRAY_TASK_ID" ]]; then
  INDEX=$((SLURM_ARRAY_TASK_ID - 1))
  while [ $INDEX -lt $((TOTAL_EXPERIMENTS-1)) ]
  do
    if [ -s "${OUTPUTS_ARRAY[$INDEX]}" ]
    then
      echo "Already computed: ${OUTPUTS_ARRAY[$INDEX]}"
    else
      eval "${COMMANDS_ARRAY[$INDEX]} > ${OUTPUTS_ARRAY[$INDEX]}"
    fi
    INDEX=$((INDEX+TOTAL_PROCESSES))
  done
else
  for i in `seq 0 $((${#COMMANDS_ARRAY[@]} - 1))`; do
    eval "${COMMANDS_ARRAY[$i]} > ${OUTPUTS_ARRAY[$i]}"
  done
fi
