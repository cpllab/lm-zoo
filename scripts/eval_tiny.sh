#!/bin/bash
TINY_ROOT="/om/group/cpl/language-models/lm-tinylstm"
python ${TINY_ROOT}/eval.py \
    --data ${TINY_ROOT}/data/ptb \
    --checkpoint ${TINY_ROOT}/model_small.pt \
    --eval_data $1 \
    --fpath $2
