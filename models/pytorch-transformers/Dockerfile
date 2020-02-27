FROM pytorch/pytorch:latest

# Root of model directory relative to the build context.
ARG MODEL_ROOT=models/pytorch-transformers
ARG COMMIT

# Set up output volume.
VOLUME /out

#RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext

# Test dependencies
RUN pip install nose rednose jsonschema
ENV NOSE_REDNOSE 1

# Download source code and pretrained model.
RUN pip install transformers

# Copy in custom file for surprisal evaluation
RUN mkdir /opt/pytorch-transformers
COPY ${MODEL_ROOT}/get_surprisals.py /opt/pytorch-transformers/get_surprisals.py
COPY ${MODEL_ROOT}/tokenizer.py /opt/pytorch-transformers/tokenizer.py
COPY ${MODEL_ROOT}/model_meta.py /opt/pytorch-transformers/model_meta.py

ENV PYTHONIOENCODING utf-8:surrogateescape

# Copy external-facing scripts
COPY ${MODEL_ROOT}/bin /opt/bin
ENV PATH "/opt/bin:${PATH}"

WORKDIR /opt/bin
