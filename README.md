______________________________________________________________________

<div align="center">

# Lightning-Hydra-DVC

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

- Run training and evaluation on Cifar10 using TIMM models with Pytorch lightning & Hydra.
- Track your data and models using dvc
- Run training and inference on Kaggle's cats and dogs dataset using Vit Transformer model. 

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
#### Dev mode

```bash
pip install -e .
```
#### Train model with default/cpu configuration

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
docker run rswain1486/emlov3-pytorchlightning-hydra sh -c "python3 src/train.py && python3 src/eval.py"

# Using volume you can mount checkpoint to host directory and run train and eval separately.
docker run --rm -t -v ${pwd}/ckpt:/workspace/ckpt rswain1486/emlov3-pytorchlightning-hydra python src/train.py
docker run --rm -t -v ${pwd}/ckpt:/workspace/ckpt rswain1486/emlov3-pytorchlightning-hydra python src/eval.py

# Post evaluation, you should see test metrics as below :
```

<div align="center">
  
<img width="316" alt="image" src="https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/e30daa20-9f61-4712-bdeb-fdf75d140703">

</div>


## How to push and pull data using DVC

```bash
# Track and update your data by creating or updating data.dvc file.
dvc add data

# Push latest data to dvc source - google drive using
dvc push -r gdrive

# Pull data tracked by dvc from source - google drive using
dvc pull -r gdrive

# To switch between versions of code and data run
git checkout master
dvc checkout

# To automate the dvc checkout everytime a git checkout is done run
dvc install

```

## Run inference on Kaggle's cats and dogs dataset
```bash
# If installed using dev mode, run infer with experiment/cat_dog_infer.yaml using
src_infer experiment=cat_dog_infer test_path=./data/PetImages_split/test/Cat/18.jpg

# If installed using requirements.txt, use
python src/infer.py experiment=cat_dog_infer test_path=./data/PetImages_split/test/Cat/18.jpg

# Predictions for Top k classes (here 2) should show as below
```

<div align="center">
  
![image](https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/8cf73be0-0fcf-4b66-9c1a-099d2c32fd05)

</div>
