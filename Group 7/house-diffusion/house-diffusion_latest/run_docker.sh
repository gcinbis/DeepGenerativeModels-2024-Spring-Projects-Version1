#!/bin/bash

docker build -t house_diffusion .

docker run -it --gpus all -v "$(pwd)":/workspace --name house_diffusion house_diffusion