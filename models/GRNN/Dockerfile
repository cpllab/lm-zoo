FROM pytorch/pytorch:latest

# Perl is required for TreeTagger tokenizer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        perl && \
        rm -rf /var/lib/apt/lists/*

# Set up output volume.
VOLUME /out

# Copy in tokenizer.
COPY tokenizer /opt/tokenizer

# Copy in source code + pretrained model.
COPY colorlessgreenRNNs /opt/colorlessgreenRNNs
ADD https://dl.fbaipublicfiles.com/colorless-green-rnns/best-models/English/hidden650_batch128_dropout0.2_lr20.0.pt /opt/colorlessgreenRNNs

# Copy in Wikipedia vocab file.
RUN mkdir -p /opt/colorlessgreenRNNs/data/wiki
COPY vocab.txt /opt/colorlessgreenRNNs/data/wiki

# Copy in custom file for surprisal evaluation
COPY evaluate_target_word_test.py /opt/colorlessgreenRNNs/src/language_models

# Copy external-facing scripts
COPY bin /opt/bin
ENV PATH "/opt/bin:${PATH}"

WORKDIR /opt/bin
