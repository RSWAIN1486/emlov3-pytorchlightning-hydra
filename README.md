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
- Use hydra multirun using joblib to train vit model on cifar10 dataset for multiple patch size using docker and view mlflow logger UI during runtime.

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


```
##### Post evaluation, you should see test metrics as below :
<div align="left">
  
<img width="316" alt="image" src="https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/e30daa20-9f61-4712-bdeb-fdf75d140703">

</div>


## How to push and pull data using DVC

```bash
# Track and update your data by creating or updating data.dvc file.
dvc add data

# To push to google drive, create folder under gdrive and add the remote to local using folder id.
dvc remote add --default gdrive gdrive://1WcXEK-HjdaQ-xZp6NOnSUqprPGhijFeE

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

```
##### Predictions for Top k classes (here 2) should show as below
<div align="left">
  
![image](https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/8cf73be0-0fcf-4b66-9c1a-099d2c32fd05)

</div>

## How to train using hydra multirun with joblib launcher (dataset cifar10, model Vit, patch size 1,2,4,8,16)
```bash
# Build Docker on local
docker build -t lightning-hydra-multiexperiments .
# or pull from Docker hub
docker pull rswain1486/lightning-hydra-experimenttracking:latest

# Run below command to start the patch size experiment using hydra joblib launcher.
# NOTE: Make sure to add port mapping from container to host if you would like to view MLflow Logger UI during runtime
# NOTE: Make sure to add volume mapping of local host directory to container workspace directory to save logs, models on local for dvc tracking.
docker run -it --expose 5000 -p 5000:5000 -v ${pwd}:/workspace --name mlflow-container lightning-hydra-experimenttracking:latest src_train -m hydra/launcher=joblib hydra.launcher.n_jobs=5 experiment=cifar10 model.patch_size=1,2,4,8,16 datamodule.num_workers=0

# Run below command to start MLFlow Logger server inside the container and open http://localhost:5000 on your browser
docker exec -it -w /workspace/logs/mlflow mlflow-container mlflow ui --host 0.0.0.0

# Post the container is exited, you can start MLFlow Logger server using and open http://localhost:5000 on your browser
cd ./logs/mlflow
mlflow ui

# Add dvc tracking to data, logs and models. (Models are saved under logs for mlflow)
dvc add data
dvc add logs
dvc config core.autostage true

git add data.dvc
git add logs.dvc

# To push to google drive, refer to the section - How to push and pull data using DVC

```

##### Multi runs in MLflow
<div align="center">
  
![image](https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/a300c844-f4e1-47ee-a892-78744545a713)

</div>

##### Scatter plot of patch_size vs val/acc in MLflow 
<div align="center">
  
![image](https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/b09698ef-84d5-48e6-b3dd-98aa9db0e713)

</div>

##### Single run directory structure under logs/mlflow saving models, metrics, metadata etc.
<div align="left">
  
![image](https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/e5354b6e-4fae-4fdf-b6de-96281fef1b24)


</div>




