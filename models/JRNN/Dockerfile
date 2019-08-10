FROM tensorflow/tensorflow:latest

# Root of model directory relative to build context.
ARG MODEL_ROOT=models/JRNN

# Set up output volume.
VOLUME /out

# Copy in source code + pretrained model.
RUN mkdir -p /opt/lm_1b
COPY ${MODEL_ROOT}/lm_1b_eval.py ${MODEL_ROOT}/data_utils.py ${MODEL_ROOT}/eval_test_google.py /opt/lm_1b/
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
        apt-get remove -y curl && apt-get autoremove -y && cd -

# Copy tokenizer.
COPY ${MODEL_ROOT}/tokenizer /opt/tokenizer

# Copy in test file.
COPY test.py /opt/test.py

# Copy external-facing scripts
COPY ${MODEL_ROOT}/bin /opt/bin
ENV PATH "/opt/bin:${PATH}"

WORKDIR /opt/bin
