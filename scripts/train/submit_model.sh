#!/bin/bash
## Expected usage: ./submit_model.sh <MODEL_KEY> <CORPUS> <SEED>

MODEL_KEY=$1
CORPUS=$2
SEED=$3
LOGS="/om/group/cpl/language-models/syntaxgym/models/training_logs"

sbatch \
    --export=MODEL_KEY=$MODEL_KEY,CORPUS=$CORPUS,SEED=$SEED,LOGS=$LOGS \
    --job-name=${MODEL_KEY}_${CORPUS}_${SEED} \
    --output ${LOGS}/${MODEL_KEY}_${CORPUS}_${SEED}.out \
    train_${MODEL_KEY}.batch