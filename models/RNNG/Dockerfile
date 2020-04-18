FROM continuumio/miniconda3

# Root of model directory relative to build context.
ARG MODEL_ROOT=models/RNNG

# Install Boost dependency.
# Need to build manually -- the standard Debian boost will pull
# in another install of Python. We'll link Boost to the existing
# Anaconda install instead.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake zlib1g-dev

# TODO Conda cmd not on path .. ? update this
ENV PATH "/opt/conda/bin:${PATH}"

# Install NLTK for tokenization and Eigen for modeling.
RUN conda install --quiet --yes nltk eigen boost \
      && conda clean --all -f -y

# Set up output volume.
VOLUME /out


# Build arguments provide SSH keys for accessing private CPL data.
ARG CPL_SSH_PRV_KEY
RUN mkdir /root/.ssh && echo "StrictHostKeyChecking no" >> /root/.ssh/config \
      && echo "$CPL_SSH_PRV_KEY" > /root/.ssh/id_rsa \
      && chmod 600 /root/.ssh/id_rsa

# Copy in source code + model parameters.
RUN git clone cpl@45.79.223.150:rnng-incremental.git /opt/rnng-incremental
RUN cd /opt/rnng-incremental && git checkout docker && cd -

# Compile source.
# NB: requires ~2 GB RAM.
WORKDIR /opt/rnng-incremental
RUN mkdir build && cd build \
  && cmake -DEIGEN3_INCLUDE_DIR=/opt/conda/include/eigen3 -DBOOST_INCLUDEDIR=/opt/conda/include/boost -DBOOST_ROOT=/opt/conda .. && make -j2

# Add parameters + resources.
RUN scp -o "StrictHostKeyChecking=no" \
      cpl@45.79.223.150:/home/cpl/rnng-incremental/{ntparse_gen_D0.3_2_256_256_16_256-pid20681.params,train_gen.oracle} /opt/rnng-incremental/
COPY ${MODEL_ROOT}/vocab.pkl /opt/rnng-incremental/

# Remove SSH information.
RUN rm -rf /root/.ssh

# Copy external-facing scripts
COPY ${MODEL_ROOT}/bin /opt/bin
COPY shared/spec /opt/bin/spec
COPY shared/unkify /opt/bin/unkify
ENV PATH "/opt/bin:${PATH}"

# Get model vocab from clusters file.
RUN cut -f2 /opt/rnng-incremental/clusters-train-berk.txt > /opt/vocab

# Copy test dependencies.
RUN pip install nose rednose jsonschema
ENV NOSE_REDNOSE 1
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

ENV LD_LIBRARY_PATH "/opt/conda/lib:${LD_LIBRARY_PATH}"

WORKDIR /opt/bin
