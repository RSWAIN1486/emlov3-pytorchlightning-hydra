import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is recommended at the top of each start file
# to make the environment more robust and consistent
#
# the line above searches for ".git" or "pyproject.toml" in present and parent dirs
# to determine the project root dir
#
# adds root dir to the PYTHONPATH (if `pythonpath=True`)
# so this file can be run from any place without installing project as a package
#
# sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
# this makes all paths relative to the project root
#
# additionally loads environment variables from ".env" file (if `dotenv=True`)
#
# you can get away without using `pyrootutils.setup_root(...)` if you:
# 1. move this file to the project root dir or install project as a package
# 2. modify paths in "configs/paths/default.yaml" to not use PROJECT_ROOT
# 3. always run this file from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import List, Tuple
from PIL import Image
import hydra
from omegaconf import DictConfig
import lightning.pytorch as L
from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import json
from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def infer(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating best model <{cfg.ckpt_path}>")
    ckpt = torch.load(cfg.ckpt_path)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    log.info(f"Loaded Model: {model}")

    categories = [
        "cat",
        "dog",
    ]
    
    transforms = T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_path = cfg.test_path
    img = Image.open(image_path)
    if img is None:
        return None
    img = transforms(img).unsqueeze(0)
    logits = model(img)
    preds = F.softmax(logits, dim=1).squeeze(0).tolist()
    out = torch.topk(torch.tensor(preds), 2)
    topk_prob  = out[0].tolist()
    topk_label = out[1].tolist()
    
    print(' \n Top k Predictions :')
    pred_json  = {categories[topk_label[i]]: topk_prob[i] for i in range(2)}
    print(json.dumps(pred_json, indent = 3))
    print('\n')
    
    return pred_json


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig) -> None:
    infer(cfg)


if __name__ == "__main__":
    main()