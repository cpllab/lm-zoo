FROM alpine AS builder

# Fetch model checkpoints
RUN apk add curl
RUN mkdir -p /opt/transformers/models/gpt2 && \
        cd /opt/transformers/models/gpt2 && \
        curl -so config.json https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json && \
        curl -so pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin && \
        curl -so vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json && \
        curl -so merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt


FROM cpllab/language-models:transformers-base

ARG MODEL_ROOT=models/gpt2

COPY --from=builder /opt/transformers/models/gpt2 /opt/transformers/models/gpt2

ENV TRANSFORMER_MODEL_TYPE gpt2
ENV TRANSFORMER_MODEL_PATH /opt/transformers/models/gpt2

# Current git commit of build repository
ARG COMMIT
# sha1 checksum of build directory
ARG FILES_SHA1

# Override model code.
COPY ${MODEL_ROOT}/get_surprisals.py /opt/transformers/get_surprisals.py
COPY ${MODEL_ROOT}/tokenizer.py /opt/transformers/tokenizer.py

# Prepare spec.
COPY ${MODEL_ROOT}/spec.template.json /tmp/spec.template.json
RUN BUILD_DATETIME="$(date)" cat /tmp/spec.template.json | \
            sed "s/<image\.sha1>/$COMMIT/g; s/<image\.files_sha1>/${FILES_SHA1}/g; s/<image\.datetime>/${BUILD_DATETIME}/g" \
            > /opt/spec.json
