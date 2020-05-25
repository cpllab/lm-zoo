FROM continuumio/miniconda3:4.8.2 as builder

# Install Boost dependency.
# Need to build manually -- the standard Debian boost will pull
# in another install of Python. We'll link Boost to the existing
# Anaconda install instead.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake zlib1g-dev

# Install NLTK for tokenization and Eigen for modeling.
# NB skip MKL (huge filesize) in favor of OpenBLAS
RUN conda install -qy --freeze-installed nomkl nltk=3.4.5 eigen=3.3.7 boost=1.67.0 \
      && conda clean --all -f -y

# Copy source code + model parameters via SSH
# Build arguments provide SSH keys for accessing private CPL data.
ARG CPL_SSH_PRV_KEY
RUN mkdir ${HOME}/.ssh && echo "StrictHostKeyChecking no" >> ${HOME}/.ssh/config \
      && echo "$CPL_SSH_PRV_KEY" > ${HOME}/.ssh/id_rsa \
      && chmod 600 ${HOME}/.ssh/id_rsa

# Copy in source code.
RUN git clone cpl@45.79.223.150:rnng-incremental.git /opt/rnng-incremental \
      && cd /opt/rnng-incremental && git checkout docker

# Add checkpoint.
ARG CHECKPOINT_PATH=cpl@45.79.223.150:/home/cpl/rnng-incremental/checkpoint
RUN mkdir -p /opt/rnng-incremental/checkpoint && \
      scp -To "StrictHostKeyChecking=no" \
        "${CHECKPOINT_PATH}/{ntparse_gen.params,train_gen.oracle,clusters.txt}" /opt/rnng-incremental/checkpoint

# Remove SSH information.
RUN rm -rf ${HOME}/.ssh

# Compile source.
# NB: requires ~2 GB RAM.
WORKDIR /opt/rnng-incremental
RUN mkdir build && cd build \
  && cmake -DEIGEN3_INCLUDE_DIR=/opt/conda/include/eigen3 -DBOOST_INCLUDEDIR=/opt/conda/include/boost -DBOOST_ROOT=/opt/conda .. && make -j2 \
  && find . -iwholename '*cmake*' -not -name CMakeLists.txt -delete



FROM continuumio/miniconda3:4.8.2

RUN apt-get install -y --no-install-recommends zlib1g

# Root of model directory relative to build context.
ARG MODEL_ROOT=models/RNNG

# Copy from previous stage
RUN rm -rf /opt/conda
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/rnng-incremental /opt/rnng-incremental

# Copy external-facing scripts and resources
COPY ${MODEL_ROOT}/bin /opt/bin
COPY ${MODEL_ROOT}/vocab.pkl /opt/rnng-incremental/
COPY shared/spec /opt/bin/spec
COPY shared/unkify /opt/bin/unkify
COPY shared/unsupported /opt/bin/get_predictions.hdf5

ENV PATH "/opt/bin:${PATH}"
ENV LMZOO_CHECKPOINT_PATH /opt/rnng-incremental/checkpoint
ENV LMZOO_VOCABULARY_PATH vocab

# Get model vocab from clusters file.
RUN cut -f2 "${LMZOO_CHECKPOINT_PATH}/clusters.txt" \
      > "${LMZOO_CHECKPOINT_PATH}/${LMZOO_VOCABULARY_PATH}"

# Copy test dependencies.
RUN pip install nose jsonschema
COPY test.py /opt/test.py

# Current git commit of build repository
ARG COMMIT
# sha1 checksum of build directory
ARG FILES_SHA1

# Prepare spec.
COPY ${MODEL_ROOT}/spec.template.json /tmp/spec.template.json
RUN BUILD_DATETIME="$(date)" cat /tmp/spec.template.json | \
            sed "s/<image\.sha1>/$COMMIT/g; s/<image\.files_sha1>/${FILES_SHA1}/g; s/<image.datetime>/${BUILD_DATETIME}/g" \
            > /opt/spec.template.json

ENV PYTHONIOENCODING utf-8

WORKDIR /opt/bin
