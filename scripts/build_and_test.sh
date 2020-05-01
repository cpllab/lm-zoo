#!/usr/bin/env bash

model_dirname=$1
target=$2

COMMIT=$(git rev-parse --verify HEAD)
CPL_SSH_PRV_KEY="$(cat ~/.ssh/id_rsa)"

model_dir=models/${model_dirname}
# Compute SHA checksum of model directory
sha_out=($(tar -c ${model_dir} | sha1sum))
FILES_SHA1=$sha_out

docker build -t $target -f ${model_dir}/Dockerfile\
        --build-arg COMMIT=$COMMIT \
        --build-arg FILES_SHA1=$FILES_SHA1 \
        --build-arg "CPL_SSH_PRV_KEY=${CPL_SSH_PRV_KEY}" \
        . \
    && docker run --rm -v `pwd`/test.py:/test.py -v `pwd`/docs/schemas:/schemas $target python /test.py
