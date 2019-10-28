#!/bin/sh
# USAGE: ./train_all.sh <CORPUS>

# Set corpus.
CORPUS=$1

# List of model keys.
MODEL_KEYS=("transxl") # "vanilla" "ordered-neurons" "transxl"

# List of seeds.
SEEDS=(0111 1025)

# Would be better to do this as a job array, but since we're starting with
# ~6 jobs, it's probably okay to have a for-loop for now...
for MODEL_KEY in "${MODEL_KEYS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "$MODEL_KEY $CORPUS $SEED"
        ./submit_model.sh $MODEL_KEY $CORPUS $SEED
    done
done