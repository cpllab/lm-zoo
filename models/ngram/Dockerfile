FROM debian:latest

ARG MODEL_ROOT=models/ngram
# Current git commit of build repository
ARG COMMIT

RUN apt update && apt install -y --no-install-recommends \
          build-essential openssh-client \
          gawk python3 python3-pip python3-setuptools
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN mkdir -p /opt/srilm

# Build arguments provide SSH keys for accessing private CPL data.
ARG CPL_SSH_PRV_KEY
RUN mkdir /root/.ssh && echo "StrictHostKeyChecking no" >> /root/.ssh/config \
      && echo "$CPL_SSH_PRV_KEY" > /root/.ssh/id_rsa \
      && chmod 600 /root/.ssh/id_rsa
RUN scp -o "StrictHostKeyChecking=no" \
      cpl@syntaxgym.org:/home/cpl/srilm/{srilm-1.7.2.tar.gz,wiki_kn_5gram.lm} /opt/srilm
RUN rm -rf /root/.ssh

WORKDIR /opt/srilm
RUN tar -xvzf srilm-1.7.2.tar.gz && \
        cp Makefile tmpf && \
        cat tmpf | awk -v pwd=`pwd` '/SRILM =/{printf("SRILM = %s\n", pwd); next;} {print;}' > Makefile && \
        make && \
        rm srilm-1.7.2.tar.gz

# Copy in test dependencies.
RUN pip3 install wheel
RUN pip3 install nose rednose jsonschema
ENV NOSE_REDNOSE 1
COPY test.py /opt/test.py

# Remove build dependencies.
RUN apt remove -y openssh-client build-essential

COPY ${MODEL_ROOT}/vocab.txt /opt/vocab.txt
COPY ${MODEL_ROOT}/get_surprisals.awk /opt/get_surprisals.awk
COPY ${MODEL_ROOT}/tokenizer /opt/tokenizer
COPY ${MODEL_ROOT}/bin /opt/bin
COPY shared/unkify /opt/bin/unkify

ENV PATH "/opt/bin:${PATH}"
ENV PYTHONIOENCODING utf-8

# Prepare spec.
COPY ${MODEL_ROOT}/spec.template.json /tmp/spec.template.json
RUN cat /tmp/spec.template.json | \
            sed "s/<image\.sha1>/$COMMIT/g; s/<image.datetime>/${BUILD_DATETIME}/g" \
            > /opt/spec.json
