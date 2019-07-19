FROM continuumio/anaconda

# Install Boost dependency.
# Need to build manually -- the standard Debian boost will pull
# in another install of Python. We'll link Boost to the existing
# Anaconda install instead.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake zlib1g-dev

# Install NLTK for tokenization and Eigen for modeling.
RUN conda install --quiet --yes nltk eigen boost \
  && conda clean --all -f -y
RUN python -m nltk.downloader punkt

# Set up output volume.
VOLUME /out

# Copy in source code + model parameters.
RUN git clone https://${CPLLAB_GITHUB_TOKEN}@github.com/cpllab/rnng-incremental.git /opt/rnng-incremental
RUN cd /opt/rnng-incremental && git checkout docker && cd -

# Compile source.
# NB: requires ~2 GB RAM.
WORKDIR /opt/rnng-incremental
RUN mkdir build && cd build \
  && cmake -DEIGEN3_INCLUDE_DIR=/opt/conda/include/eigen3 -DBOOST_INCLUDEDIR=/opt/conda/include/boost -DBOOST_ROOT=/opt/conda .. && make -j2

# Add parameters + resources.
RUN scp -o "StrictHostKeyChecking=no" \
      cpl@45.79.223.150:/home/cpl/rnng-incremental/{ntparse_gen_D0.3_2_256_256_16_256-pid20681.params,train_gen.oracle} /opt/rnng-incremental/
COPY vocab.pkl /opt/rnng-incremental/

# Copy external-facing scripts
COPY bin /opt/bin
ENV PATH "/opt/bin:${PATH}"

# Copy test dependencies.
RUN pip install nose
COPY test.py /opt/test.py

ENV LD_LIBRARY_PATH "/opt/conda/lib:${LD_LIBRARY_PATH}"

WORKDIR /opt/bin
