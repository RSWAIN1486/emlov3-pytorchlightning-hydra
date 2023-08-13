______________________________________________________________________

<div align="center">

# Extensive MLOps 

![image](https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/8be39374-ad41-418a-bf5e-4071820224a7)


<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

# Overview
This repository is an implementation of all the sessions covered as part of EMLO V3.0 course. [Course Syllabus](https://docs.google.com/document/u/1/d/e/2PACX-1vT2LBXrRYbg7NM4Q6j2rMSokiC3Vt_rMk4E8k_vTUSJt_4HAEFfSsO_DAmOCl6nV3fRbwmSdadbBfTL/pub)

# Main Technologies used
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - a lightweight PyTorch wrapper for high-performance AI research. Think of it as a framework for organizing your PyTorch code.

- [Docker](https://docs.docker.com/get-started/overview/) - an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly

- [Hydra](https://github.com/facebookresearch/hydra) - a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

- [DVC](https://github.com/iterative/dvc) - a command line tool to help you develop reproducible machine learning projects by versioning your data and models.

- [MLFlow](https://github.com/mlflow/mlflow) - a platform to streamline machine learning development, including tracking experiments, packaging code into reproducible runs, and sharing and deploying models

- [Optuna](https://github.com/optuna/optuna) - an automatic hyperparameter optimization software framework, particularly designed for machine learning

- [TorchScript](https://pytorch.org/docs/stable/jit.html) - a way to create serializable and optimizable models from PyTorch code.

- [TorchTrace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) - a way to trace a function and return an executable or ScriptFunction that will be optimized using just-in-time compilation.

- [Gradio](https://github.com/gradio-app/gradio) - an open-source Python library that is used to quickly build machine learning and data science demos and web applications.

- [FastAPI](https://github.com/tiangolo/fastapi) - a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.

- [Locust](https://github.com/locustio/locust) - an easy to use, scriptable and scalable performance testing tool.

- [AWS ECS](https://docs.aws.amazon.com/ecs/index.html) - a highly scalable, fast, container management service that makes it easy to run, stop, and manage Docker containers on a cluster of Amazon EC2 instances.

- [AWS Fargate](https://docs.aws.amazon.com/AmazonECS/latest/userguide/what-is-fargate.html) - a technology that you can use with Amazon ECS to run containers without having to manage servers or clusters of Amazon EC2 instances.

- [AWS ECR](https://docs.aws.amazon.com/AmazonECS/latest/userguide/ecr-repositories.html) - a managed AWS Docker registry service that can be used with ECS.

# Table of Contents

- [Session 4  : How to run on local](#how-to-run-on-local)
- [Session 4  : Run Training and Evaluation using Docker on Cifar10 using TIMM models with Pytorch lightning and Hydra](#run-training-and-evaluation-using-docker-on-cifar10-using-timm-models-with-pytorch-lightning-and-hydra)
- [Session 5  : How to Push and Pull Data and Models using DVC](#how-to-push-and-pull-data-and-models-using-dvc)
- [Session 5  : Run Inference on Kaggle's cats and dogs dataset using Vit Transformer model](#run-training-and-inference-on-kaggle-cats-and-dogs-dataset-using-vit-transformer-model)
- [Session 6  : Train using Hydra multirun with Joblib launcher (Dataset cifar10, Model Vit, patch_size 1,2,4,8,16) and View Runs in MLflow](#train-using-hydra-multirun-with-joblib-launcher-and-view-runs-in-mlflow)
- [Session 7  : HyperParameter Optimization using Optuna and Hydra Multirun (Dataset HarryPotter, Model GPT)](#hyperparameter-optimization-using-optuna-and-hydra-multirun)
- [Session 8  : Gradio Demo with Torch Script model (Dataset Cifar10, Model Vit)](#gradio-demo-with-torchscript-model)
- [Session 8  : Gradio Demo with Torch Trace model (Dataset HarryPotter, Model GPT)](#gradio-demo-with-torch-trace-model)
- [Session 9  : Gradio Demo with GPT Traced model (Dataset HarryPotter, Model GPT) on AWS using ECR, ECS, S3](#gradio-demo-with-gpt-traced-model)
- [Session 10 : FastAPI Demo with Docker](#fastapi-demo-with-docker)
- [Session 11 : Deploy CLIP with Docker and FastAPI on ECS Fargate (Multi Replicas) and Stress Test with Locust](#deploy-clip-with-docker-and-fastapi-on-ecs-fargate-and-stress-test-with-locust)

## How to Run on Local

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
#### Dev Mode

```bash
pip install -e .
```
#### Train Model with default/cpu configuration

```bash
# train on CPU
python src/train.py trainer=cpu
python src/eval.py

# You can override any parameter from command line like this
python src/train.py trainer.max_epochs=20 data.batch_size=64
```


## Run Training and Evaluation using Docker on Cifar10 using TIMM models with Pytorch lightning and Hydra

```bash
# Build Docker on local
docker build -t emlov3-pytorchlightning-hydra .
# or pull from Docker hub
docker pull rswain1486/emlov3-pytorchlightning-hydra:latest

# Since checkpoint will not be persisted between container runs if train and eval are run separately, use below command to run together. 
docker run rswain1486/emlov3-pytorchlightning-hydra sh -c "python3 src/train.py andand python3 src/eval.py"

# Using volume you can mount checkpoint to host directory and run train and eval separately.
docker run --rm -t -v ${pwd}/ckpt:/workspace/ckpt rswain1486/emlov3-pytorchlightning-hydra python src/train.py
docker run --rm -t -v ${pwd}/ckpt:/workspace/ckpt rswain1486/emlov3-pytorchlightning-hydra python src/eval.py


```
##### Post evaluation, you should see test metrics as below :
<div align="left">
  
<img width="316" alt="image" src="https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/e30daa20-9f61-4712-bdeb-fdf75d140703">

</div>


## How to Push and Pull Data and Models using DVC

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

### Run Training and Inference on Kaggle cats and dogs dataset using Vit Transformer model
```bash
# Training
# If installed using dev mode, run infer with experiment/cat_dog_infer.yaml using
src_train experiment=cat_dog trainer.max_epochs=1 datamodule.batch_size=64 datamodule.num_workers=0

# If installed using requirements.txt, use
python src/train.py experiment=cat_dog trainer.max_epochs=1 datamodule.batch_size=64 datamodule.num_workers=0

# Inference
# If installed using dev mode, run infer with experiment/cat_dog_infer.yaml using
src_infer experiment=cat_dog_infer test_path=./data/PetImages_split/test/Cat/18.jpg

# If installed using requirements.txt, use
python src/infer.py experiment=cat_dog_infer test_path=./data/PetImages_split/test/Cat/18.jpg

```
##### Predictions for Top k classes (here 2) should show as below
<div align="left">
  
![image](https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/8cf73be0-0fcf-4b66-9c1a-099d2c32fd05)

</div>

## Train using Hydra multirun with Joblib launcher and View Runs in MLflow
```bash
# Build Docker on local
docker build -t lightning-hydra-multiexperiments .
# or pull from Docker hub
docker pull rswain1486/lightning-hydra-experimenttracking:latest

# Run below command to start the patch size experiment using hydra joblib launcher.
# NOTE: Make sure to add port mapping from container to host if you would like to view MLflow Logger UI during runtime
# NOTE: Make sure to add volume mapping of local host directory to container workspace directory to save logs, models on local for dvc tracking.
docker run -it --expose 5000 -p 5000:5000 -v ${pwd}:/workspace --name mlflow-container lightning-hydra-experimenttracking:latest \
src_train -m hydra/launcher=joblib hydra.launcher.n_jobs=5 experiment=cifar10 model.patch_size=1,2,4,8,16 datamodule.num_workers=0

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

##### View Multi runs in MLflow
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

## HyperParameter Optimization using Optuna and Hydra Multirun
<a href="https://colab.research.google.com/github/RSWAIN1486/emlov3-pytorchlightning-hydra/blob/main/examples/HParams_Optimization_Lightning_Hydra_Optuna_GPT_HarryPotter.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
```bash
# Find the Best Learning Rate and Batch size using Lightning Tuner
src_train -m tuner=True train=False test=False datamodule.num_workers=2 experiment=harrypotter

# Run Hyperparameter Search using Optuna using Hydra config file
src_train -m test=False datamodule.num_workers=2 experiment=harrypotter hparams_search=harrypotter_optuna

# Load MLFlow logger UI to compare HyperParameter Experiments
cd logs/mlflow
mlflow ui

# Run Training for n epochs with Best HyperHarameters
src_train -m tuner=True test=False trainer.max_epochs=10 datamodule.num_workers=2 experiment=harrypotter datamodule.block_size=8 \
model.block_size=8 model.net.block_size=8 model.net.n_embed=256 model.net.n_heads=8 model.net.drop_p=0.15 model.net.n_decoder_blocks=4
```

##### Scatter plot of different HyperParameters across Experiments in MLflow
<div align="left">

![image](https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/960b2748-a4fe-483d-8660-98dc81488c43)


</div>

#### Generate text using GPT for Harry Potter using best hyperparams

<div align="left">

![image](https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/075fbbb9-fffd-4270-9e61-77f9f2fc368c)


</div>

## Gradio Demo with TorchScript model

```bash
# Install in dev mode
pip install -e .

# Train the Vit model on Cifar10 and save as TorchScript model. Set save_torchscript to True in configs/train.yaml
src_train experiment=cifar10_jit save_torchscript=True

# Infer on a test image using Torchscript model
src_infer_jit_script_vit test_path=./test/0000.jpg

# Launch Gradio Demo for Cifar10 at port 8080 and open http://localhost:8080/
# NOTE: Set the ckpt_path and labels_path in configs/infer_jit_script_vit.yaml
src_demo_jit_script_vit

# Build and Launch Gradio Demo using Docker. This should launch demo at http://localhost:8080/. Ensure to expose the port in docker-compose/ DockerFile.demo
docker compose  -f docker-compose.yml up --build demo_cifar_gradio

# Launch Gradio demo by pulling from Dockerhub
docker run -p 8080:8080 rswain1486/gradio-cifar10-demo:latest

# To stop the demo, if the Ctrl + C does not work, use
docker stop $(docker ps -aq)

```
#### Gradio UI for Cifar10

<div align="left">

![image](https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/af46c1f1-a702-41c0-bbfe-45ccb97f718f)

</div>

## Gradio Demo with Torch Trace model

<a href="https://colab.research.google.com/github/RSWAIN1486/emlov3-pytorchlightning-hydra/blob/main/examples/GPT_HarryPotter_TorchTrace.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

```bash
# Install in dev mode
pip install -e .

# Train the GPT model on HarryPotter dataset and save as Torch traced model. Set save_torchtrace to True in configs/train.yaml
src_train -m experiment=harrypotter_jit.yaml test=False trainer.max_epochs=20 trainer.accelerator=gpu save_torchtrace=True paths.ckpt_jittrace_save_path=ckpt/gpt_torch_traced.pt

# Generate text using Torch traced model
src_infer_jit_trace_gpt ckpt_path=ckpt/gpt_torch_traced.pt input_txt='Avada Kedavra'

# Launch Gradio demo
src_demo_jit_trace_gpt ckpt_path=ckpt/gpt_torch_traced.pt
```
#### Gradio UI for generating Harry Potter text using GPT

<div align="left">

![image](https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/6f5b76fb-38b4-4a53-b51f-88d33b109f73)


</div>

## Gradio Demo with GPT Traced model
```bash

# Build and Launch Gradio Demo using Docker. This should launch demo at http://localhost:80/
docker compose  -f docker-compose.yml up --build demo_gpt_gradio

# Test using
python3 src/gradio/test_demo_jit_script_gpt.py

# If aws is configured, push the model to S3 using. Set the bucket_name and model_file_path.
python3 src/aws/push_model_S3.py

# To push your docker image to ECR, run below commands
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ecr repo uri>
docker build -t <repo-name> .
docker tag <repo-name>:latest <ecr repo uri>/<repo-name>:latest
docker push <ecr repo uri>/<repo-name>:latest

```

## FastAPI Demo with Docker
```bash

# Build and launch FastAPI using Docker. This should launch demo at http://localhost:8080/docs or http://<ec2-public-ip>/docs
# for GPT
docker-compose  -f docker-compose.yml up --build demo_gpt_fastapi

# for VIT
docker-compose  -f docker-compose.yml up --build demo_vit_fastapi

# To generate a log file with individual and average response time for 100 api requests.
# for GPT. Set the server url and log file path accordingly.
python3 src/fastapi/gpt/test_api_calls_gpt.py

# for VIT. Set the server url, input image file and log file path accordingly.
python3 src/fastapi/vit/test_api_calls_vit.py

```
#### Average response time for GPT

<div align="left">

![image](https://github.com/RSWAIN1486/emlov3-fastapi/assets/48782471/39b7e473-588a-4956-b249-88aaca5bf8b0)

</div>

#### Average response time for VIT

<div align="left">

![image](https://github.com/RSWAIN1486/emlov3-fastapi/assets/48782471/6c564894-1ac6-4963-bace-d291aff77823)

</div>

#### CPU usage with 2 workers for GPT

<div align="left">

![image](https://github.com/RSWAIN1486/emlov3-fastapi/assets/48782471/2bff0f77-f7cd-4a6f-8e84-552824b78650)

</div>

## Deploy CLIP with Docker and FastAPI on ECS Fargate and Stress Test with Locust
```bash

# Build and launch CLIP using FastAPI. This should launch demo at http://localhost:80/ or http://<ec2-public-ip>:80/docs

docker-compose  -f docker-compose.yml up --build demo_clip_fastapi

# If deployed using docker image on AWS ECS Fargate using load balancer, it should launch at http://<DNS-of-load-balancer>/docs

# To start locust server and start swarming. By default, server should start at http://localhost:8089/ or http://<ec2-public-ip>:8089/
python3 src/clip/locust_stress_test_clip.py

# To install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.4/install.sh | bash
nvm install 16
nvm use 16
cd src/clip/clip-frontend
npx create-next-app@latest clip-frontend

# To start the clip front end, edit the page.tsx under clip-frontend/app accordingly and run
npm run dev

```
#### Locust Stress Test with CLIP deployed on ECS Fargate

<div align="left">

![clip_locust750](https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/7b40164d-a8b1-474d-8570-03863ae623b0)

</div>

#### CLIP Deployed on frontend

<div align="left">

![image](https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra/assets/48782471/540755fb-ec62-4b0b-8afc-204927d5d7da)

</div>
