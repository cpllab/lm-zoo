FROM cpllab/language-models:pytorch-transformers

RUN pip install sentencepiece

RUN mkdir -p /opt/pytorch-transformers/models/xlnet
RUN cd /opt/pytorch-transformers/models/xlnet && \
        curl -so spiece.model https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-spiece.model && \
        curl -so pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-pytorch_model.bin && \
        curl -so config.json https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-config.json

ENV PYTORCH_TRANSFORMER_MODEL_TYPE xlnet
ENV PYTORCH_TRANSFORMER_MODEL_PATH /opt/pytorch-transformers/models/xlnet
