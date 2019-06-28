FROM pytorch/pytorch:latest

# Set up output volume.
VOLUME /out

RUN mkdir -p /opt/bin

# Copy in source code + pretrained model.
COPY colorlessgreenRNNs /opt/colorlessgreenRNNs
ADD https://dl.fbaipublicfiles.com/colorless-green-rnns/best-models/English/hidden650_batch128_dropout0.2_lr20.0.pt /opt/colorlessgreenRNNs

# Copy in Wikipedia vocab file.
RUN mkdir -p /opt/colorlessgreenRNNs/data/wiki
COPY vocab.txt /opt/colorlessgreenRNNs/data/wiki

# Copy in custom file for surprisal evaluation
COPY evaluate_target_word_test.py /opt/colorlessgreenRNNs/src/language_models

# Copy external-facing scripts
COPY *.sh /opt/bin

WORKDIR /opt/bin
