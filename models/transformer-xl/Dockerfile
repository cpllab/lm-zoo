FROM pytorch/pytorch:latest

# Set up output volume.
VOLUME /out

#RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext

# Download source code and pretrained model.
RUN pip install pytorch-pretrained-bert
RUN mkdir -p /opt/transformer-xl/transfo-xl-wt103
RUN wget -O /opt/transformer-xl/transfo-xl-wt103/pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-pytorch_model.bin
RUN wget -O /opt/transformer-xl/transfo-xl-wt103/vocab.bin https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-vocab.bin
RUN wget -O /opt/transformer-xl/transfo-xl-wt103/config.json https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-config.json

# Copy in custom file for surprisal evaluation
COPY eval_transfo.py /opt/transformer-xl

# Test dependencies
RUN pip install nose rednose
ENV NOSE_REDNOSE 1
COPY test.py /opt/test.py

# Copy external-facing scripts
COPY bin /opt/bin
ENV PATH "/opt/bin:${PATH}"

WORKDIR /opt/bin
