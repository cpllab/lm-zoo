#!/bin/bash
JRNN_ROOT="/om/group/cpl/language-models/jrnn"
python ${JRNN_ROOT}/lm_1b/eval_test_google.py \
    --pbtxt ${JRNN_ROOT}/data/graph-2016-09-10.pbtxt \
    --ckpt '/om/group/cpl/language-models/jrnn/data/ckpt-*' \
    --vocab_file ${JRNN_ROOT}/data/vocab-2016-09-10.txt \
    --input_file $1 \
    --output_file $2
