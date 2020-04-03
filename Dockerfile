FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.8
ARG WITH_TORCHVISION=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    unzip \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
    /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
    /opt/conda/bin/conda install -y libgcc flatbuffers && \
    /opt/conda/bin/conda clean -afy --all && \
    /opt/conda/bin/pip uninstall -y dask

ENV PATH /opt/conda/bin:$PATH
# This must be done before pip so that requirements.txt is available
WORKDIR /opt/pytorch
COPY . .

RUN git submodule sync && git submodule update --init --recursive
RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v . && \
    git clone https://github.com/pytorch/vision.git && \
    cd vision && \
    pip install -v .

RUN pip install -U https://ray-wheels.s3-us-west-2.amazonaws.com/latest/ray-0.9.0.dev0-cp36-cp36m-manylinux1_x86_64.whl
RUN pip install -U boto3
# We install this after the latest wheels -- this should not override the latest wheels.
# Needed to run Tune example with a 'plot' call - which does not actually render a plot, but throws an error.
RUN apt-get install -y zlib1g-dev libgl1-mesa-dev
# The following is needed to support TensorFlow 1.14
RUN conda remove -y --force wrapt
RUN pip install gym[atari]==0.10.11 opencv-python-headless tensorflow lz4 keras pytest-timeout smart_open torch torchvision dm_tree
RUN pip install --upgrade bayesian-optimization
RUN pip install --upgrade hyperopt==0.1.2
RUN pip install ConfigSpace==0.4.10
RUN pip install --upgrade sigopt nevergrad scikit-optimize hpbandster lightgbm xgboost tensorboardX
RUN pip install -U mlflow
RUN pip install -U pytest-remotedata>=0.3.1

# RUN mkdir -p /root/.ssh/

# We port the source code in so that we run the most up-to-date stress tests.
ADD ray.tar /ray
ADD git-rev /ray/git-rev
RUN python /ray/python/ray/setup-dev.py --yes

WORKDIR /ray
RUN chmod -R a+w .
