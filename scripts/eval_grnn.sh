#!/bin/sh
GRNN_ROOT="/om/group/cpl/language-models/colorlessgreenRNNs"
python ${GRNN_ROOT}/src/language_models/evaluate_target_word_test.py \
    --data ${GRNN_ROOT}/data/wiki \
    --checkpoint ${GRNN_ROOT}/hidden650_batch128_dropout0.2_lr20.0.pt \
    --prefixfile $1 \
    --outf $2 \
    --surprisalmode True
