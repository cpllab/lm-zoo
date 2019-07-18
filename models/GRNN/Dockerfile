FROM pytorch/pytorch:latest

# Perl is required for TreeTagger tokenizer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        perl && \
        rm -rf /var/lib/apt/lists/*

# Add test dependencies.
# TODO: better to handle in a docker-compose setup, I think
RUN pip install nose rednose
ENV NOSE_REDNOSE 1

# Set up output volume.
VOLUME /out

# Copy in tokenizer.
COPY tokenizer /opt/tokenizer

# Copy in source code + pretrained model.
RUN git clone git://github.com/facebookresearch/colorlessgreenRNNs /opt/colorlessgreenRNNs \
        && cd /opt/colorlessgreenRNNs \
        && git checkout 4ffcabc991c866608aeed2ba35059a458ff2845f
RUN curl -so /opt/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt https://dl.fbaipublicfiles.com/colorless-green-rnns/best-models/English/hidden650_batch128_dropout0.2_lr20.0.pt

# Copy in Wikipedia vocab file.
RUN mkdir -p /opt/colorlessgreenRNNs/data/wiki
COPY vocab.txt /opt/colorlessgreenRNNs/data/wiki/

# Copy in custom file for surprisal evaluation
COPY evaluate_target_word_test.py /opt/colorlessgreenRNNs/src/language_models/

# Copy in image test file.
COPY test.py /opt/test.py

# Copy external-facing scripts
COPY bin /opt/bin
ENV PATH "/opt/bin:${PATH}"

WORKDIR /opt/bin
