#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="uranium",
    version="0.0.1",
    description="PyTorch Lightning Project Setup using Hydra with TIMM template.",
    author="",
    author_email="",
    url="https://github.com/RSWAIN1486/emlov3-pytorchlightning-hydra",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "uranium_train = uranium.train:main",
            "uranium_eval = uranium.eval:main",
        ]
    },
)