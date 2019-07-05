FROM continuumio/anaconda3

# Install Boost dependency.
# Need to build manually -- the standard Debian boost will pull
# in another install of Python. We'll link Boost to the existing
# Anaconda install instead.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake
# RUN mkdir -p /usr/src && cd /usr/src && \
#  wget --no-verbose https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.gz && \
#  tar xzf boost_1_65_1.tar.gz && \
#  cd boost_1_65_1 && \
#  ./bootstrap.sh --with-python=$(which python3) && \
#  ./b2 install && \
#  ldconfig && \
#  cd / && rm -rf /usr/src/*

# Install NLTK for tokenization and Eigen for modeling.
RUN conda install --quiet --yes nltk eigen boost \
  && conda clean --all -f -y
RUN python -m nltk.downloader punkt

# Set up output volume.
VOLUME /out

# Copy in source code + model parameters.
COPY rnng /opt/rnng
COPY clusters-train-berk.txt /opt/rnng/clusters-train-berk.txt
COPY gdown.pl .
RUN perl gdown.pl https://drive.google.com/open?id=1E0lrR5sypgiYsYQmIdYN6TKVds22ApXa /opt/rnng/ntparse_gen_D0.3_2_256_256_16_256-pid64164.params && rm gdown.pl

# Compile model.
WORKDIR /opt/rnng
RUN mkdir build && cd build \
  && cmake -DEIGEN3_INCLUDE_DIR=/opt/conda/include/eigen3 -DBOOST_INCLUDEDIR=/opt/conda/include/boost -DBOOST_ROOT=/opt/conda .. && make -j1

# Copy external-facing scripts
COPY bin /opt/bin
ENV PATH "/opt/bin:${PATH}"

WORKDIR /opt/bin
