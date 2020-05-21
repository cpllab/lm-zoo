FROM alpine AS builder

## TODO: Build code and dependencies, and/or fetch model checkpoints.
## We recommend fetching checkpoints in this build stage, since we don't want
## the final image to carry along unnecessary packages like curl/wget. Also,
## you can fetch checkpoints securely over SSL in this stage using private keys
## which won't be included in the final build. See e.g. the RNNG Dockerfile in
## LM Zoo for an example.
#
# RUN apk add curl
RUN mkdir -p /opt/mylm
RUN mkdir -p /opt/mylm/checkpoint
# RUN cd /opt/mylm/checkpoint && \
#           curl -so model.ckpt https://me.com/mylm/model.ckpt &&
#           curl -so vocab.txt https://me.com/mylm/vocab.txt


## TODO: Pick a base image. Explicitly specify an image tag when possible (e.g.
## `tensorflow/tensorflow:1.15.2`).
##
## Popular base images:
# FROM pytorch/pytorch
# FROM tensorflow/tensorflow
# FROM continuumio/miniconda3
FROM continuumio/miniconda3

# Root of model directory relative to build context.
ARG MODEL_ROOT=models/_template

## TODO: Install runtime dependencies.
## Maybe copy over dependencies from build image,
# COPY --from=builder /usr/local/my_dependency /usr/local/my_dependency
## Or install via package manager
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         perl && \
#         rm -rf /var/lib/apt/lists/*

## Add test dependencies. These are required for all images -- otherwise tests
## will fail.
RUN conda install -y numpy nomkl && conda clean -tipsy
RUN pip install nose jsonschema

## TODO: Fetch/copy model dependencies.
COPY --from=builder /opt/mylm /opt/mylm
COPY ${MODEL_ROOT}/vocab.txt /opt/mylm/checkpoint/vocab.txt

## TODO: Copy in custom code wrapping the language model
COPY ${MODEL_ROOT}/get_word_surprisals.py /opt/mylm/get_word_surprisals.py

## Copy external-facing scripts.
COPY ${MODEL_ROOT}/bin /opt/bin
## LM Zoo provides standard scripts for some of the binaries. `unkify` output,
## for example, can be computed automatically from the output of `tokenize` and
## the language model's spec. The shared `unkify` script does just this for
## you.
COPY shared/unkify /opt/bin/unkify
## The LM Zoo shared spec script will automatically insert the language model's
## vocabulary into a template spec. Handy!
COPY shared/spec /opt/bin/spec
## If your image doesn't support certain features, copy in the shared script
## ``unsupported`` for each of these features. This makes things easily
## detectable from the LM Zoo API.
COPY shared/unsupported /opt/bin/get_predictions.hdf5

ENV PATH "/opt/bin:${PATH}"

## Set the path to the language model's checkpoint data. The image user can
## override this environment variable, in which case the corresponding new
## checkpoint should be loaded. (This is a requirement of models which support
## the `mount_checkpoint` feature.)
ENV LMZOO_CHECKPOINT_PATH /opt/mylm/checkpoint

## State the path (relative to the checkpoint path) of the model's vocabulary.
## It isn't necessary to do this, but in doing so we can use the shared LM Zoo
## `spec` binary to auto-generate a spec output containing our vocabulary.
ENV LMZOO_VOCABULARY_PATH vocab.txt

## Fix I/O encoding.
ENV PYTHONIOENCODING utf-8

## Prepare language model specification. We'll take the template specification
## and add information provided from a few Docker build args.
# Current git commit of build repository
ARG COMMIT
# sha1 checksum of build directory
ARG FILES_SHA1
# Prepare spec.
COPY ${MODEL_ROOT}/spec.template.json /opt/spec.template.json
RUN BUILD_DATETIME="$(date)" sed -i "s/<image\.sha1>/$COMMIT/g; s/<image\.files_sha1>/${FILES_SHA1}/g; s/<image\.datetime>/${BUILD_DATETIME}/g" /opt/spec.template.json

WORKDIR /opt/bin
