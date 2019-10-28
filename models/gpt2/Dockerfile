FROM cpllab/language-models:pytorch-transformers

RUN mkdir -p /opt/pytorch-transformers/models/gpt2
RUN cd /opt/pytorch-transformers/models/gpt2 && \
        curl -so config.json https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json && \
        curl -so pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin && \
        curl -so vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json && \
        curl -so merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

ENV PYTORCH_TRANSFORMER_MODEL_TYPE gpt2
ENV PYTORCH_TRANSFORMER_MODEL_PATH /opt/pytorch-transformers/models/gpt2
