FROM tensorflow/tensorflow:1.15.2

# Root of model directory relative to build context.
ARG MODEL_ROOT=models/JRNN

# Set up output volume.
VOLUME /out

# Copy in source code + pretrained model.
RUN mkdir -p /opt/lm_1b
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN cd /opt/lm_1b && \
        curl -sO http://download.tensorflow.org/models/LM_LSTM_CNN/graph-2016-09-10.pbtxt && \
        curl -sO http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-base && \
        curl -sO http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-char-embedding && \
        curl -sO http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-lstm && \
        curl -sO http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax0 && \
        curl -sO http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax1 && \
        curl -sO http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax2 && \
        curl -sO http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax3 && \
        curl -sO http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax4 && \
        curl -sO http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax5 && \
        curl -sO http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax6 && \
        curl -sO http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax7 && \
        curl -sO http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax8 && \
        curl -sO http://download.tensorflow.org/models/LM_LSTM_CNN/vocab-2016-09-10.txt && \
        sed -i '/^!!!MAXTERMID$/d' vocab-2016-09-10.txt && \
        apt-get remove -y curl && apt-get autoremove -y && cd -
COPY ${MODEL_ROOT}/lm_1b_eval.py ${MODEL_ROOT}/data_utils.py ${MODEL_ROOT}/eval_test_google.py /opt/lm_1b/

# Copy tokenizer.
COPY ${MODEL_ROOT}/tokenizer /opt/tokenizer

# Install runtime dependencies.
RUN pip install h5py

# Copy in test dependencies.
RUN pip install nose jsonschema
COPY test.py /opt/test.py

# Copy external-facing scripts
COPY ${MODEL_ROOT}/bin /opt/bin
COPY shared/unkify /opt/bin
ENV PATH "/opt/bin:${PATH}"

# Current git commit of build repository
ARG COMMIT
# sha1 checksum of build directory
ARG FILES_SHA1

# Prepare spec.
COPY ${MODEL_ROOT}/spec.template.json /tmp/spec.template.json
RUN BUILD_DATETIME="$(date)" cat /tmp/spec.template.json | \
            sed "s/<image\.sha1>/$COMMIT/g; s/<image\.files_sha1>/${FILES_SHA1}/g; s/<image.datetime>/${BUILD_DATETIME}/g" \
            > /opt/spec.json

WORKDIR /opt/bin
