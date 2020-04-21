FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.7
ARG WITH_TORCHVISION=1
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         wget && \
     rm -rf /var/lib/apt/lists/*


RUN wget -O ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
# This must be done before pip so that requirements.txt is available

WORKDIR /opt
RUN git clone https://github.com/pytorch/pytorch.git
    
WORKDIR /opt/pytorch

RUN git checkout tags/v1.4.1 &&git submodule sync && git submodule update --init --recursive
RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v .

RUN if [ "$WITH_TORCHVISION" = "1" ] ; then git clone https://github.com/pytorch/vision.git && cd vision && pip install -v . ; else echo "building without torchvision" ; fi

WORKDIR /workspace
RUN chmod -R a+w .