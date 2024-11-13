import os
from argparse import ArgumentParser
from easydict import EasyDict as edict
from ruamel.yaml import YAML

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as torchDist
from torch.distributed import get_world_size

from mdhpnet.utils.train_pipelines import distTrainPipeline
from mdhpnet.utils.math import setRandomSeed
from mdhpnet.utils.factory import buildModel, buildDataset, buildCriterion, buildOptimizer
from mdhpnet.utils.simple_logger import initLoggers

def create_dist_dataloader(cfgs: dict | edict) -> DataLoader:
    """
    Create a distributed dataloader.

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
        Distributed dataloader.
    """
    cfgs = edict(cfgs)
    dataset = buildDataset(cfgs.Dataset)
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=get_world_size(),
        rank=torchDist.get_rank(),
        shuffle=True,
    )
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=cfgs.batch_size,
        shuffle=cfgs.shuffle,
        num_workers=cfgs.num_workers,
        drop_last=True,
    )
    return dataloader

def main(cfgs: dict | edict):
    cfgs = edict(cfgs)

    setRandomSeed(cfgs.RandomSeed)

    # Initialize the distributed training environment
    rank = int(os.environ["LOCAL_RANK"])
    torchDist.init_process_group(
        backend="nccl",
        rank=rank,
        init_method="env://",
    )

    setRandomSeed(cfgs.RandomSeed, rank)

    train_dataloader = create_dist_dataloader(cfgs.TrainDataLoader)
    val_dataloader = create_dist_dataloader(cfgs.ValDataLoader)

    model = buildModel(cfgs.Model)
    optimizer = buildOptimizer(cfgs.Optimizer, model)
    criterion = buildCriterion(cfgs.Criterion)

    train_cfgs = cfgs.TrainPipeline
    distTrainPipeline(
        rank=rank,
        world_size=get_world_size(),
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=train_cfgs.n_epochs,
        ckpt_save_dir=train_cfgs.ckpt_save_dir,
        ckpt_save_interval=train_cfgs.ckpt_save_interval,
        statistics_save_dir=train_cfgs.statistics_save_dir,
        logger_name=train_cfgs.logger_name,
    )

    # Destroy the process group
    torchDist.destroy_process_group()


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