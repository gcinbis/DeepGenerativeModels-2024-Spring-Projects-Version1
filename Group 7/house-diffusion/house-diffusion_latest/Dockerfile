FROM pytorchlightning/pytorch_lightning:base-cuda-py3.11-torch2.3-cuda12.1.0

ENV TZ=Europe/Istanbul
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    zip \
    nano \
    wget \
    ffmpeg \
    libsm6 \
    libxext6

RUN pip install --no-cache-dir \
    opencv-python \
    networkx \
    shapely \
    jupyter 

WORKDIR /workspace