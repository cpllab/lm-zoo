FROM continuumio/miniconda3:4.8.2-alpine AS builder

USER root
RUN apk add git openssh-client build-base

ARG CPL_SSH_PRV_KEY
RUN mkdir ${HOME}/.ssh && echo "StrictHostKeyChecking no" >> ${HOME}/.ssh/config \
      && echo "$CPL_SSH_PRV_KEY" > ${HOME}/.ssh/id_rsa \
      && chmod 600 ${HOME}/.ssh/id_rsa
RUN mkdir -p /opt/srilm \
        && scp -To "StrictHostKeyChecking=no" \
            cpl@syntaxgym.org:/home/cpl/srilm/{srilm-1.7.2.tar.gz,wiki_kn_5gram.lm} /opt/srilm \
        && mkdir /opt/srilm/checkpoint \
        && mv /opt/srilm/wiki_kn_5gram.lm /opt/srilm/checkpoint/model.lm
RUN rm -rf ${HOME}/.ssh

RUN cd /opt/srilm && tar -xvzf srilm-1.7.2.tar.gz && \
        cp Makefile tmpf && \
        cat tmpf | awk -v pwd=`pwd` '/SRILM =/{printf("SRILM = %s\n", pwd); next;} {print;}' > Makefile && \
        make && make cleanest && \
        rm srilm-1.7.2.tar.gz


FROM continuumio/miniconda3:4.8.2-alpine

ARG MODEL_ROOT=models/ngram

ENV PATH "/opt/conda/bin:$PATH"
COPY --from=builder /opt/srilm /opt/srilm

# Runtime dependencies
USER root
RUN apk add perl bash libstdc++ gawk libgomp

# Copy in test dependencies.
RUN conda install -qy --freeze-installed numpy nomkl \
        && pip install nose jsonschema
COPY test.py /opt/test.py

ENV LMZOO_CHECKPOINT_PATH /opt/srilm/checkpoint
ENV LMZOO_VOCABULARY_PATH vocab

COPY ${MODEL_ROOT}/vocab.txt "$LMZOO_CHECKPOINT_PATH/$LMZOO_VOCABULARY_PATH"
COPY ${MODEL_ROOT}/get_surprisals.awk /opt/get_surprisals.awk
COPY ${MODEL_ROOT}/tokenizer /opt/tokenizer
COPY ${MODEL_ROOT}/bin /opt/bin
COPY shared/unkify /opt/bin/unkify
COPY shared/spec /opt/bin/spec
COPY shared/unsupported /opt/bin/get_predictions.hdf5

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
            > /opt/spec.template.json
