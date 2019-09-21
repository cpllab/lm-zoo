FROM cpllab/language-models:pytorch-transformers

# Root of model directory relative to the build context.
ARG MODEL_ROOT=models/roberta

RUN mkdir -p /opt/pytorch-transformers/models/roberta
RUN cd /opt/pytorch-transformers/models/roberta && \
        curl -so pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin && \
        curl -so vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json && \
        curl -so merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt && \
        curl -so config.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json

ENV PYTORCH_TRANSFORMER_MODEL_TYPE roberta
ENV PYTORCH_TRANSFORMER_MODEL_PATH /opt/pytorch-transformers/models/roberta

# Copy in custom file for surprisal evaluation
COPY ${MODEL_ROOT}/get_surprisals.py /opt/pytorch-transformers/get_surprisals.py
