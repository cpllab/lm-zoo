#!/bin/sh
TRANS_ROOT="/om/user/group/cpl/transformer-xl"
python ${TRANS_ROOT}/eval_transfo.py \
    -model ${TRANS_ROOT}/model \
    -inputf $1 \
    -outputf $2 \
    --cuda
