# Multi Agent Car Racing Environment
<img width="1200" height="631" alt="Multi-agent CarRacing environment" src="https://github.com/user-attachments/assets/6306134e-f56c-47b2-aaf6-9e1dff2bf240" />

This repository contains a multi-agent version of the original [CarRacing environment](https://gymnasium.farama.org/environments/box2d/car_racing/).

## Installation
### Swig
Swig is required to install the *box2d* feature for gymnasium environments.
Swig is the abbreviation of *Simplified Wrapper and Interface Generator*, it can give script language such as python the ability to invoke C and C++ libraries interface method indirectly.

[](https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/) (On Windows make sure to download **swig**win-xxx which includes a prebuild executable)

## Training
It is recommended to run the training in a docker container on a machine with one available GPU. The WANDB_API_KEY environment variable needs to be set. The following commands start the training process:
### Single Agent
`docker build -t rllib-car-racing-training . && docker run --gpus=1 --shm-size=9.98gb --env WANDB_API_KEY=\"$WANDB_API_KEY\" --detach rllib-car-racing-training python train-single-ppo.py`

### Multi Agent
`docker build -t rllib-car-racing-training . && docker run --gpus=1 --shm-size=9.98gb --env WANDB_API_KEY=\"$WANDB_API_KEY\" --detach rllib-car-racing-training python train-multi-ppo.py`

### Multi Agent with Curriculum
`docker build -t rllib-car-racing-training . && docker run --gpus=1 --shm-size=9.98gb --env WANDB_API_KEY=\"$WANDB_API_KEY\" --detach rllib-car-racing-training python train-multi-ppo-curriculum.py`
