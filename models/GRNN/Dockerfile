FROM pytorch/pytorch:latest

# Root of model directory relative to build context.
ARG MODEL_ROOT=models/GRNN
# Current git commit of build repository
ARG COMMIT

# Prepare spec.
COPY spec.template.json /tmp/spec.template.json
RUN cat /tmp/spec.template.json && \
            sed "s/<image\.sha1>/$COMMIT/g; s/<image.datetime>/${BUILD_DATETIME}" \
            > /opt/spec.json

# Perl is required for TreeTagger tokenizer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        perl && \
        rm -rf /var/lib/apt/lists/*

# Add test dependencies.
# TODO: better to handle in a docker-compose setup, I think
RUN pip install nose rednose
ENV NOSE_REDNOSE 1

# Copy in tokenizer.
COPY ${MODEL_ROOT}/tokenizer /opt/tokenizer

# Copy in source code.
RUN git clone git://github.com/facebookresearch/colorlessgreenRNNs /opt/colorlessgreenRNNs \
        && cd /opt/colorlessgreenRNNs \
        && git checkout 4ffcabc991c866608aeed2ba35059a458ff2845f
# Copy in pretrained model.
RUN curl -so /opt/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt https://dl.fbaipublicfiles.com/colorless-green-rnns/best-models/English/hidden650_batch128_dropout0.2_lr20.0.pt

# Copy in Wikipedia vocab file.
RUN mkdir -p /opt/colorlessgreenRNNs/data/wiki
COPY ${MODEL_ROOT}/vocab.txt /opt/colorlessgreenRNNs/data/wiki/

# Copy in custom file for surprisal evaluation
COPY ${MODEL_ROOT}/evaluate_target_word_test.py /opt/colorlessgreenRNNs/src/language_models/

# Copy in shared tests.
COPY test.py /opt/test.py

# Copy external-facing scripts
COPY ${MODEL_ROOT}/bin /opt/bin
ENV PATH "/opt/bin:${PATH}"
ENV PYTHONIOENCODING utf-8

WORKDIR /opt/bin
