
import os
from os import path as osp
from argparse import ArgumentParser
from easydict import EasyDict as edict
from ruamel.yaml import YAML
import logging
import random

import torch
from torch.utils.data import DataLoader
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from mdhpnet.utils.math import setRandomSeed

from mdhpnet.utils.test_pipelines import testPipeline
from mdhpnet.utils.factory import buildModel, buildDataset
from mdhpnet.utils.simple_logger import initLoggers



def create_dataloader(cfgs: dict | edict) -> DataLoader:
    """
    Create a dataloader.

    Parameters
    ----------
    cfgs : dict | EasyDict
        Configuration dictionary.
        Must contain the following keys:
        data_dir: str
        batch_size: int
        shuffle: bool
        num_workers: int

    Returns
    -------
    DataLoader
    """
    cfgs = edict(cfgs)
    dataset = buildDataset(cfgs.Dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfgs.batch_size,
        shuffle=cfgs.shuffle,
        num_workers=cfgs.num_workers,
        pin_memory=cfgs.pin_memory,
        drop_last=True,
    )
    return dataloader

def main(cfgs: dict | edict):
    cfgs = edict(cfgs)

    setRandomSeed(cfgs.RandomSeed)

    test_dataloader = create_dataloader(cfgs.TestDataLoader)
    model = buildModel(cfgs.Model)
    test_cfgs = cfgs.TestPipeline

    testPipeline(
        device=test_cfgs.device,
        test_dataloader=test_dataloader,
        model=model,
        statistics_save_dir=test_cfgs.statistics_save_dir,
        logger_name=test_cfgs.logger_name,
    )

if __name__ == "__main__":
    initLoggers()
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--cfg_file",
        type=str,
        dest="cfg",
        required=True,
        help="Path to the config file (YAML)",
    )
    args = parser.parse_args()
    cfg = YAML().load(open(args.cfg))
    main(cfg)