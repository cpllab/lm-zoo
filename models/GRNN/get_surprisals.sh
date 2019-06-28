#!/bin/sh
GRNN_ROOT="/opt/colorlessgreenRNNs"
python ${GRNN_ROOT}/src/language_models/evaluate_target_word_test.py \
    --data ${GRNN_ROOT}/data/wiki \
    --checkpoint ${GRNN_ROOT}/hidden650_batch128_dropout0.2_lr20.0.pt \
    --prefixfile $1 \
    --outf /out/surprisals.tsv \
    --surprisalmode True
