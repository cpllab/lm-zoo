FROM continuumio/miniconda3:4.10.3

# Root of model directory relative to the build context.
ARG MODEL_ROOT=models/transformers-base
ARG COMMIT

RUN conda install -y -c pytorch pytorch=1.9 torchvision cpuonly && \
        conda clean -ya

# Test dependencies
RUN pip install nose jsonschema

# Runtime dependencies
RUN pip install transformers h5py

# Copy in custom file for surprisal evaluation
RUN mkdir /opt/transformers
COPY ${MODEL_ROOT}/get_surprisals.py /opt/transformers/get_surprisals.py
COPY ${MODEL_ROOT}/tokenizer.py /opt/transformers/tokenizer.py

ENV PYTHONIOENCODING utf-8:surrogateescape
# open issue with pytorch https://github.com/pytorch/pytorch/issues/37377
ENV MKL_SERVICE_FORCE_INTEL=1

# Copy external-facing scripts
COPY ${MODEL_ROOT}/bin /opt/bin
ENV PATH "/opt/bin:${PATH}"

WORKDIR /opt/bin
