#!/bin/bash
RNNG_ROOT="/om/group/cpl/language-models/rnng-incremental"
export LD_LIBRARY_PATH="${RNNG_ROOT}/dependencies/boost_1_68_0/lib"
${RNNG_ROOT}/build/nt-parser/nt-parser-gen \
    --dynet-mem 2000 \
    -x -T ${RNNG_ROOT}/train_gen.oracle \
    -v $1 \
    -f $2 \
    --clusters ${RNNG_ROOT}/clusters-train-berk.txt \
    --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 \
    -m ${RNNG_ROOT}/ntparse_gen_D0.3_2_256_256_16_256-pid20681.params
