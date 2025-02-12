#!/bin/bash

WORKSPACE=$(pwd)/../apps/tvlt
CONTAINER_NAME=nvcr.io/nvidia/pytorch
CONTAINER_VERSION=22.08-py3
NAME=dropinf_tvlt

# Two seperate volumes to avoid git error:
## stderr: 'error: unknown option `cached'
docker run --cap-add SYS_ADMIN --shm-size 8G -d -v $WORKSPACE:/workspace -v $(pwd)/../:/dropinf -v /tmp/nvidia-mps:/tmp/nvidia-mps --gpus all --rm --pid=host --ipc=host --net=host --name=${NAME} --interactive --tty ${CONTAINER_NAME}:${CONTAINER_VERSION}
docker exec ${NAME} bash -c "pip install torchaudio"
docker exec ${NAME} bash -c "pip install pytorch_lightning"
docker exec ${NAME} bash -c "pip install sacred"
docker exec ${NAME} bash -c "pip install transformers"
docker exec ${NAME} bash -c "pip install einops"
docker exec ${NAME} bash -c "pip install timm"
docker exec ${NAME} bash -c "pip install decord"
docker exec ${NAME} bash -c "pip install librosa"
docker exec ${NAME} bash -c "pip install audiosegment"
docker exec ${NAME} bash -c "pip install moviepy"
docker exec ${NAME} bash -c "pip install ffmpeg-python"
docker exec ${NAME} bash -c "pip install stable_baselines3"
docker exec ${NAME} bash -c "pip install huggingface_sb3"
docker exec ${NAME} bash -c "pip install setuptools"
docker exec ${NAME} bash -c "pip install tensorboard"
docker exec ${NAME} bash -c "pip install gekko"
docker exec ${NAME} bash -c "apt update"
docker exec ${NAME} bash -c "DEBIAN_FRONTEND=noninteractive apt install ffmpeg -y"
docker exec ${NAME} bash -c "export PYTHONPATH=${PYTHONPATH}:/dropinf:/dropinf/dropinf"

docker exec -it ${NAME} bash
