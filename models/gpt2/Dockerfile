FROM cpllab/language-models:pytorch-transformers

ARG MODEL_ROOT=models/gpt2
ARG COMMIT

RUN mkdir -p /opt/pytorch-transformers/models/gpt2
RUN cd /opt/pytorch-transformers/models/gpt2 && \
        curl -so config.json https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json && \
        curl -so pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin && \
        curl -so vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json && \
        curl -so merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

ENV PYTORCH_TRANSFORMER_MODEL_TYPE gpt2
ENV PYTORCH_TRANSFORMER_MODEL_PATH /opt/pytorch-transformers/models/gpt2

# Prepare spec.
COPY ${MODEL_ROOT}/spec.template.json /tmp/spec.template.json
RUN cat /tmp/spec.template.json | \
            sed "s/<image\.sha1>/$COMMIT/g; s/<image.datetime>/${BUILD_DATETIME}/g" \
            > /opt/spec.json

