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
import hydra
from omegaconf import DictConfig
import torch
import tiktoken
from src import utils


log = utils.get_pylogger(__name__)

@utils.task_wrapper

def infer(cfg: DictConfig) -> Tuple[str, str]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path
    assert cfg.input_txt

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        torch.manual_seed(42)

    log.info(f"Instantiating traced model <{cfg.ckpt_path}>")
    loaded_model = torch.jit.load(cfg.ckpt_path)

    # encoder
    cl100k_base = tiktoken.get_encoding("cl100k_base")

    # In production, load the arguments directly instead of accessing private attributes
    # See openai_public.py for examples of arguments for specific encodings
    encoder = tiktoken.Encoding(
        # If you're changing the set of special tokens, make sure to use a different name
        # It should be clear from the name what behaviour to expect.
        name="cl100k_im",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<|im_start|>": 100264,
            "<|im_end|>": 100265,
                }
            )

    input_enc = torch.tensor(encoder.encode(cfg.input_txt))
    with torch.no_grad():
        out_gen = loaded_model.model.generate(input_enc.unsqueeze(0).long(), max_new_tokens=256)
    decoded = encoder.decode(out_gen[0].cpu().numpy().tolist())

    print(' \n \n <------------- Harry Potter Generated Text -------------> \n \n')
    print(decoded)
    print('\n \n <------------- End of Tokens -------------> \n \n')

    return decoded, '' 


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer_jit_trace_gpt.yaml")
def main(cfg: DictConfig) -> None:
    result_str, _ = infer(cfg)
    
if __name__ == "__main__":
    main()