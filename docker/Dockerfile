# get the development image from nvidia cuda 12.1
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

LABEL name="unique3d" maintainer="unique3d"

# create workspace folder and set it as working directory
RUN mkdir -p /workspace
WORKDIR /workspace

# update package lists and install git, wget, vim, libegl1-mesa-dev, and libglib2.0-0
RUN apt-get update && apt-get install -y build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip git-lfs

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev cmake curl mesa-utils-extra
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ENV PYOPENGL_PLATFORM=egl

# install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# update PATH environment variable
ENV PATH="/workspace/miniconda3/bin:${PATH}"

# initialize conda
RUN conda init bash

# create and activate conda environment
RUN conda create -n unique3d python=3.10 && echo "source activate unique3d" > ~/.bashrc
ENV PATH /workspace/miniconda3/envs/unique3d/bin:$PATH

RUN conda install Ninja
RUN conda install cuda -c nvidia/label/cuda-12.1.0 -y

RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 xformers triton --index-url https://download.pytorch.org/whl/cu121
RUN pip install diffusers==0.27.2

RUN git clone --depth 1 https://huggingface.co/spaces/Wuvin/Unique3D

# change the working directory to the repository

WORKDIR /workspace/Unique3D
# other dependencies
RUN pip install -r requirements.txt

RUN pip install nvidia-pyindex

RUN pip install --upgrade nvidia-tensorrt

RUN pip install spaces

