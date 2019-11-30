FROM cpllab/language-models:ordered-neurons

# Root of model directory relative to build context.
ARG SUBMODEL_ROOT=models/ordered-neurons-cased-ptb

# Build arguments provide SSH keys for accessing private CPL data.
ARG CPL_SSH_PRV_KEY
RUN mkdir /root/.ssh && echo "StrictHostKeyChecking no" >> /root/.ssh/config \
      && echo "$CPL_SSH_PRV_KEY" > /root/.ssh/id_rsa \
      && chmod 600 /root/.ssh/id_rsa

# Copy in model parameters.
ARG CHECKPOINT_NAME=jennhu-ptb-20191108
RUN scp -o "StrictHostKeyChecking=no" \
      cpl@45.79.223.150:/home/cpl/ordered-neurons/models/$CHECKPOINT_NAME/{model.pt,corpus.data} \
      /opt/Ordered-Neurons

# Remove SSH information.
RUN rm -rf /root/.ssh

# Copy in custom file for surprisal evaluation
COPY ${SUBMODEL_ROOT}/bin/tokenize /opt/bin/tokenize
COPY ${SUBMODEL_ROOT}/get_surprisals.py /opt/Ordered-Neurons/get_surprisals.py

WORKDIR /opt/bin
