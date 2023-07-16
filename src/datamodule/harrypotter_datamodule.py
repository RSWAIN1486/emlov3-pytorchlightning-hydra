import os
import shutil
from git import Repo
import glob
from pathlib import Path
import tiktoken
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import  LightningDataModule
from typing import Any, Optional
import requests


class HarryPotterDataset(Dataset):
    def __init__(self, data_dir="./data", txt_file="harrypotter.txt", block_size=8, download=True):
        super().__init__()

        self.block_size = block_size
        self.data_dir = data_dir

        file_path = Path(f"{data_dir}/{txt_file}")
        if download:
            Path(self.data_dir).mkdir(exist_ok=True)
            file_path.unlink(missing_ok=True)

            download_links = [
            "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%201%20-%20The%20Philosopher's%20Stone.txt",
            "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%202%20-%20The%20Chamber%20of%20Secrets.txt",
            "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%203%20-%20The%20Prisoner%20of%20Azkaban.txt",
            "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%204%20-%20The%20Goblet%20of%20Fire.txt",
            "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%205%20-%20The%20Order%20of%20the%20Phoenix.txt",
            "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%206%20-%20The%20Half%20Blood%20Prince.txt",
            "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%207%20-%20The%20Deathly%20Hallows.txt",
            ]
            
            

            for link in download_links:
                r = requests.get(link, allow_redirects=True)
                r.encoding = 'utf8'
                with open(file_path, 'a', encoding='utf8') as f:
                    f.write(r.text)
        else:
            print(f'Data already at {data_dir}')

        with (file_path).open(encoding='utf8') as f:
            self.text = f.read()

        cl100k_base = tiktoken.get_encoding("cl100k_base")

        # In production, load the arguments directly instead of accessing private attributes
        # See openai_public.py for examples of arguments for specific encodings
        self.encoder = tiktoken.Encoding(
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

        self.data = np.array(self.encoder.encode(self.text))

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = torch.from_numpy(
            (self.data[idx:idx + self.block_size]).astype(np.int64)
        )

        y = torch.from_numpy(
            (self.data[idx+1:idx+1+self.block_size]).astype(np.int64)
        )

        return x, y


class HarryPotterDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        txt_file: str = "harrypotter.txt",
        download: bool = True,
        train_ratio=0.7,
        batch_size: int = 64,
        num_workers: int = 0,
        block_size = 8,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def prepare_data(self):
        HarryPotterDataset(data_dir=self.hparams.data_dir, txt_file=self.hparams.txt_file, download=self.hparams.download, block_size=self.hparams.block_size)

    def setup(self, stage: Optional[str] = None):
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            dataset = HarryPotterDataset(data_dir=self.hparams.data_dir, txt_file=self.hparams.txt_file, download=self.hparams.download, block_size=self.hparams.block_size)

            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=[self.hparams.train_ratio, (1-self.hparams.train_ratio)],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "harrypotter.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
