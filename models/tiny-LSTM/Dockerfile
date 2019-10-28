FROM pytorch/pytorch:latest

# Root of model directory relative to build context.
ARG MODEL_ROOT=models/tiny-LSTM

# Install NLTK for tokenization.
RUN conda install --quiet --yes nltk \
      && conda clean --all -f -y

# Install ssh.
RUN apt-get update && apt-get install -y openssh-server

# Add test dependencies.
# TODO: better to handle in a docker-compose setup, I think
RUN pip install nose rednose
ENV NOSE_REDNOSE 1

# Set up output volume.
VOLUME /out

# Copy in shared tests.
COPY test.py /opt/test.py

# Build arguments provide SSH keys for accessing private CPL data.
ARG CPL_SSH_PRV_KEY
RUN mkdir /root/.ssh && echo "StrictHostKeyChecking no" >> /root/.ssh/config \
      && echo "$CPL_SSH_PRV_KEY" > /root/.ssh/id_rsa \
      && chmod 600 /root/.ssh/id_rsa

# Copy in source code + model parameters.
RUN git clone cpl@syntaxgym.org:tinylstm.git /opt/tinylstm
RUN cd /opt/tinylstm 

# Add parameters + resources.
RUN scp -o "StrictHostKeyChecking=no" \
      cpl@syntaxgym.org:tinylstm/{ptb.pt,model_small.pt} /opt/tinylstm/
COPY ${MODEL_ROOT}/vocab.pkl /opt/tinylstm/
# COPY ${MODEL_ROOT}/get_tokens.py /opt/tinylstm/

# Remove SSH information.
RUN rm -rf /root/.ssh

# Copy external-facing scripts
COPY ${MODEL_ROOT}/bin /opt/bin
ENV PATH "/opt/bin:${PATH}"
ENV PYTHONIOENCODING utf-8

WORKDIR /opt/bin
