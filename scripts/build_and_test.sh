#!/usr/bin/env bash

model_dir=$1
tag=$2

CPL_SSH_PRV_KEY="$(cat ~/.ssh/id_rsa)"

docker build -t cpllab/language-models:$tag -f models/${model_dir}/Dockerfile --build-arg "CPL_SSH_PRV_KEY=${CPL_SSH_PRV_KEY}" . \
    && docker run --rm -v `pwd`/test.py:/test.py -v `pwd`/docs/schemas:/schemas cpllab/language-models:${tag} python /test.py
