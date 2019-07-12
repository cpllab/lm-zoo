FROM tensorflow/tensorflow:latest

# Set up output volume.
VOLUME /out

# Copy in source code + pretrained model.
RUN mkdir -p /opt/lm_1b
COPY lm_1b_eval.py data_utils.py eval_test_google.py /opt/lm_1b/
ADD http://download.tensorflow.org/models/LM_LSTM_CNN/graph-2016-09-10.pbtxt /opt/lm_1b
ADD http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-base /opt/lm_1b
ADD http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-char-embedding /opt/lm_1b
ADD http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-lstm /opt/lm_1b
ADD http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax0 /opt/lm_1b
ADD http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax1 /opt/lm_1b
ADD http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax2 /opt/lm_1b
ADD http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax3 /opt/lm_1b
ADD http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax4 /opt/lm_1b
ADD http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax5 /opt/lm_1b
ADD http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax6 /opt/lm_1b
ADD http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax7 /opt/lm_1b
ADD http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax8 /opt/lm_1b
ADD http://download.tensorflow.org/models/LM_LSTM_CNN/vocab-2016-09-10.txt /opt/lm_1b

# Copy tokenizer.
COPY tokenizer /opt/tokenizer

# Copy in test file.
COPY test.py /opt/test.py

# Copy external-facing scripts
COPY bin /opt/bin
ENV PATH "/opt/bin:${PATH}"

WORKDIR /opt/bin
