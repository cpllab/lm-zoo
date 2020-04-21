FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

# Root of model directory relative to build context.
ARG MODEL_ROOT=models/ordered-neurons

# RUN conda install pytorch=0.4.1 -c pytorch
RUN pip install -q nltk

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client

# Add test dependencies.
RUN pip install nose rednose jsonschema
ENV NOSE_REDNOSE 1

# Copy in source code.
RUN git clone git://github.com/yikangshen/Ordered-Neurons /opt/Ordered-Neurons \
        && cd /opt/Ordered-Neurons \
        && git checkout 46d63cd

# Build arguments provide SSH keys for accessing private CPL data.
ARG CPL_SSH_PRV_KEY
RUN mkdir /root/.ssh && echo "StrictHostKeyChecking no" >> /root/.ssh/config \
      && echo "$CPL_SSH_PRV_KEY" > /root/.ssh/id_rsa \
      && chmod 600 /root/.ssh/id_rsa

# Copy in model parameters.
ARG CHECKPOINT_NAME=fredashi-20190919
RUN scp -o "StrictHostKeyChecking=no" \
      cpl@45.79.223.150:/home/cpl/ordered-neurons/models/$CHECKPOINT_NAME/{model.pt,corpus.data} \
      /opt/Ordered-Neurons

# Cache vocab list for easy access from `spec`, etc.
RUN PYTHONPATH="/opt/Ordered-Neurons:$PYTHONPATH" python \
                -c 'import torch; corpus = torch.load("/opt/Ordered-Neurons/corpus.data"); \
                    f = open("/opt/vocab", "w"); f.write("\n".join(corpus.dictionary.word2idx.keys()));'

# Remove SSH information.
RUN rm -rf /root/.ssh
# Remove build dependencies.
RUN apt-get remove -y git openssh-client && apt-get -y autoremove && rm -rf /var/lib/apt/lists/*

# Copy in custom file for surprisal evaluation
COPY ${MODEL_ROOT}/get_surprisals.py /opt/Ordered-Neurons/get_surprisals.py

# Copy in shared tests.
COPY test.py /opt/test.py

# Copy external-facing scripts
COPY ${MODEL_ROOT}/bin /opt/bin
COPY shared/unkify /opt/bin/unkify
COPY shared/spec /opt/bin/spec
ENV PATH "/opt/bin:${PATH}"

# Current git commit of build repository
ARG COMMIT
# sha1 checksum of build directory
ARG FILES_SHA1

# Prepare spec.
COPY ${MODEL_ROOT}/spec.template.json /tmp/spec.template.json
RUN export BUILD_DATETIME="$(date)"
RUN cat /tmp/spec.template.json | \
            sed "s/<image\.sha1>/$COMMIT/g; s/<image\.files_sha1>/${FILES_SHA1}/g; s/<image\.datetime>/${BUILD_DATETIME}/g" \
            > /opt/spec.template.json

WORKDIR /opt/bin
