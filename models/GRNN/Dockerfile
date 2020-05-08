# Drop SSH private key in a dummy build stage
FROM alpine:latest AS builder
ARG CPL_SSH_PRV_KEY


FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

# Root of model directory relative to build context.
ARG MODEL_ROOT=models/GRNN

# Perl is required for TreeTagger tokenizer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        perl && \
        rm -rf /var/lib/apt/lists/*

# Add runtime dependencies.
RUN pip install h5py

# Add test dependencies.
RUN pip install nose jsonschema

# Copy in tokenizer.
COPY ${MODEL_ROOT}/tokenizer /opt/tokenizer

# Copy in source code.
RUN git clone git://github.com/facebookresearch/colorlessgreenRNNs /opt/colorlessgreenRNNs \
        && cd /opt/colorlessgreenRNNs \
        && git checkout 4ffcabc991c866608aeed2ba35059a458ff2845f
# Copy in pretrained model.
RUN curl -so /opt/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt https://dl.fbaipublicfiles.com/colorless-green-rnns/best-models/English/hidden650_batch128_dropout0.2_lr20.0.pt

# Copy in Wikipedia vocab file.
RUN mkdir -p /opt/colorlessgreenRNNs/data/wiki
COPY ${MODEL_ROOT}/vocab.txt /opt/colorlessgreenRNNs/data/wiki/

# Copy in custom file for surprisal evaluation
COPY ${MODEL_ROOT}/evaluate_target_word_test.py /opt/colorlessgreenRNNs/src/language_models/

# Copy external-facing scripts
COPY ${MODEL_ROOT}/bin /opt/bin
COPY shared/unkify /opt/bin/unkify
ENV PATH "/opt/bin:${PATH}"
ENV PYTHONIOENCODING utf-8

# Current git commit of build repository
ARG COMMIT
# sha1 checksum of build directory
ARG FILES_SHA1

# Prepare spec.
COPY ${MODEL_ROOT}/spec.template.json /tmp/spec.template.json
RUN BUILD_DATETIME="$(date)" cat /tmp/spec.template.json | \
            sed "s/<image\.sha1>/$COMMIT/g; s/<image\.files_sha1>/${FILES_SHA1}/g; s/<image\.datetime>/${BUILD_DATETIME}/g" \
            > /opt/spec.json

WORKDIR /opt/bin
