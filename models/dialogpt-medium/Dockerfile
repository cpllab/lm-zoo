# This file is adapted from github.com/cpllab/lm-zoo/blob/master/models/gpt2/Dockerfile
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
RUN apt-get -y update \
    && apt-get install -y curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Root of model directory relative to the build context.

ARG MODEL_ROOT=models/dialogpt-medium
ARG COMMIT
# Test dependencies
RUN pip install nose jsonschema

# Runtime dependencies
RUN pip install sentencepiece==0.1.91 transformers==3.0.2 h5py

# Copy in custom file for surprisal evaluation
COPY ${MODEL_ROOT}/get_surprisals.py /opt/dialogpt-medium/get_surprisals.py
COPY ${MODEL_ROOT}/tokenizer.py /opt/dialogpt-medium/tokenizer.py


# Download the model files
RUN curl https://s3.amazonaws.com/models.huggingface.co/bert/microsoft/DialoGPT-medium/config.json -so /opt/dialogpt-medium/config.json
RUN curl https://cdn.huggingface.co/microsoft/DialoGPT-medium/pytorch_model.bin -so /opt/dialogpt-medium/pytorch_model.bin
RUN curl https://cdn.huggingface.co/microsoft/DialoGPT-medium/vocab.json -so /opt/dialogpt-medium/vocab.json
RUN curl https://cdn.huggingface.co/microsoft/DialoGPT-medium/merges.txt -so /opt/dialogpt-medium/merges.txt

# Copy spec template
COPY ${MODEL_ROOT}/spec.template.json /opt/spec.json

ENV PYTHONIOENCODING utf-8:surrogateescape
# open issue with pytorch https://github.com/pytorch/pytorch/issues/37377
ENV MKL_SERVICE_FORCE_INTEL=1

# Copy external-facing scripts
COPY ${MODEL_ROOT}/bin /opt/bin
ENV PATH "/opt/bin:${PATH}"

WORKDIR /opt/bin
