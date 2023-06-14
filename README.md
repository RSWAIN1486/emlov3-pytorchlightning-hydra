______________________________________________________________________

<div align="center">

# Lightning-Hydra-Timm-Cifar10

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

Run training and evaluation on Cifar10 using TIMM models with Pytorch lightning & Hydra.

## How to run on local

### Installation

#### Pip

```bash
# clone project
git clone https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra.git
cd emlov3-pytorchlightning-hydra

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```


Train model with default/cpu configuration

```bash
# train on CPU
python src/train.py trainer=cpu
python src/eval.py

# You can override any parameter from command line like this
python src/train.py trainer.max_epochs=20 data.batch_size=64
```


## How to run using Docker

```bash
# Build Docker on local
docker build -t emlov3-pytorchlightning-hydra .
# or pull from Docker hub
docker pull rswain1486/emlov3-pytorchlightning-hydra:latest

# Since checkpoint will not be persisted between container runs if train and eval are run separately, use below command to run together. 
# Using volume is another option which will be added later.
docker run rswain1486/emlov3-pytorchlightning-hydra sh -c "python3 src/train.py && python3 src/eval.py"

```

