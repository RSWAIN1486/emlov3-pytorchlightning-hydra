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
import torch
import json
from src import utils
from torchvision import transforms
from torch.nn import functional as F

log = utils.get_pylogger(__name__)
categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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
        torch.manual_seed(42)

    log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)

    image_path = cfg.test_path
    image = Image.open(image_path)
    if image is None:
        return None

    predict_transform = transforms.Compose(
                                [
                                    transforms.Resize((32, 32)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ]
                            )
    image_tensor = predict_transform(image)
    batch_image_tensor = torch.unsqueeze(image_tensor, 0)
    logits = model(batch_image_tensor)

    preds = F.softmax(logits, dim=1).squeeze(0).tolist()
    out = torch.topk(torch.tensor(preds), len(categories))
    topk_prob  = out[0].tolist()
    topk_label = out[1].tolist()

    print(' \n Top k Predictions :')
    pred_json  = {categories[topk_label[i]]: topk_prob[i] for i in range(10)}
    print(json.dumps(pred_json, indent = 3))
    print('\n')
    
    return pred_json, {}    


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer_jit_script_vit.yaml")
def main(cfg: DictConfig) -> None:
    result_dict, _ = infer(cfg)


if __name__ == "__main__":
    main()