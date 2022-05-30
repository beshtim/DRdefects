ARG CUDA_VERSION=11.4.2
ARG OS_VERSION=18.04

FROM nvcr.io/nvidia/cuda:11.4.2-cudnn8-devel-ubuntu18.04

ENV TRT_VERSION 8.2.1.8

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y  \
    git wget sudo build-essential \
    python3 python3-setuptools python3-pip python3-dev python3-tk \
    ffmpeg libsm6 libxext6
RUN ln -svf /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade --force pip

# TensorRT
RUN v="${TRT_VERSION%.*}-1+cuda${CUDA_VERSION%.*}" &&\
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub &&\
    apt-get update &&\
    sudo apt-get install libnvinfer8=${v} libnvonnxparsers8=${v} libnvparsers8=${v} libnvinfer-plugin8=${v} \
        libnvinfer-dev=${v} libnvonnxparsers-dev=${v} libnvparsers-dev=${v} libnvinfer-plugin-dev=${v} \
        python3-libnvinfer=${v}

# create a non-root user
ARG USER_ID=1000
ARG USER=appuser
RUN useradd -m --no-log-init --system --uid ${USER_ID} ${USER} -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ${USER}
WORKDIR /home/${USER}
ENV PATH="/home/${USER}/.local/bin:${PATH}"

# Install dependencies
RUN pip install --user torch torchvision 
RUN pip install --user opencv-python scipy colormath scikit-learn imutils PyYAML pandas pycuda

WORKDIR /defectdetector
COPY . .

ENV LANG C.UTF-8

ENTRYPOINT [ "sudo", "python", "main.py" ] 