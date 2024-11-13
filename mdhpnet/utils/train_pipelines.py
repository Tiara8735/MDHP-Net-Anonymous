import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as torchDist

import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import sys
from typing import Tuple

# [TODO][Enhancement] @tiara8735
# |- Add non-distributed training pipeline.


def distTrainOneEpoch(
    model: Module,
    train_dataloader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    rank: int,
    world_size: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for data, labels in train_dataloader:
        data = model.module.preprocess(data, rank)
        labels = labels.long().to(rank)
        optimizer.zero_grad()
        preds = model(data)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_labels = preds.argmax(dim=1)
            total_correct += (pred_labels == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()

    torchDist.all_reduce(
        torch.tensor(total_loss, device=rank), op=torchDist.ReduceOp.SUM
    )
    torchDist.all_reduce(
        torch.tensor(total_correct, device=rank), op=torchDist.ReduceOp.SUM
    )
    torchDist.all_reduce(
        torch.tensor(total_samples, device=rank), op=torchDist.ReduceOp.SUM
    )

    train_loss = total_loss / world_size
    train_acc = total_correct / total_samples

    return train_loss, train_acc


def validate(
    model: Module,
    val_dataloader: DataLoader,
    criterion: Module,
    rank: int,
):
    if rank != 0:
        return None, None

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in val_dataloader:
            data = model.module.preprocess(data, rank)
            labels = labels.long().to(rank)
            preds = model(data)
            loss = criterion(preds, labels)

            total_loss += loss.item()
            pred_labels = preds.argmax(dim=1)
            total_correct += (pred_labels == labels).sum().item()
            total_samples += labels.size(0)

    val_loss = total_loss / len(val_dataloader)
    val_acc = total_correct / total_samples
    return val_loss, val_acc


def distTrainPipeline(
    rank: int,
    world_size: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: Module,
    criterion: Module,
    optimizer: Optimizer,
    n_epochs: int,
    ckpt_save_dir: str = None,
    ckpt_save_interval: int = 1,
    statistics_save_dir: str = None,
    logger_name: str = "TRAIN",
):
    """
    Distributed training pipeline for MDHP models.

    Parameters
    ----------
    rank : int
        Current process rank.
    world_size : int
        Total number of processes.
    train_dataloader : DataLoader
        Training dataloader.
    val_dataloader : DataLoader
        Validation dataloader.
    model : Module
        Model to train.
    criterion : Module
        Loss function.
    optimizer : Optimizer
        Optimizer.
    n_epochs : int
        Number of total epochs.
    ckpt_save_dir : str, optional
        Directory to save the model checkpoints, by default None.
    ckpt_save_interval : int, optional
        Interval of epochs to save the model, by default 1.
    statistics_save_dir : str, optional
        Directory to save the statistics (loss & acc), by default None.
    loggerName : str, optional
        Logger name, by default "TRAIN".
        You should write the logger configuration in "./configs/logger.yml".
    """
    logger = logging.getLogger(logger_name)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    best = {
        "val_acc": 0,
        "epoch": 0,
        "ckpt": None,
    }

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    n_seq = model.module.n_seq
    model_name = model.module.__class__.__name__
    supposed_save_dir = osp.join(".", "results", "train", f"{model_name}_N{n_seq}")
    if statistics_save_dir is None:
        statistics_save_dir = osp.join(supposed_save_dir, "statistics")
    os.makedirs(statistics_save_dir, exist_ok=True)
    if ckpt_save_dir is None:
        ckpt_save_dir = osp.join(supposed_save_dir, "ckpt")
    os.makedirs(ckpt_save_dir, exist_ok=True)

    # >>>>>>>>>> Start training >>>>>>>>>>
    for epoch in range(n_epochs):
        if rank == 0 and sys.stdout.isatty():
            train_dataloader = tqdm(
                train_dataloader, desc=f"Training epoch: {epoch + 1}/{n_epochs}"
            )

        # Synchronize all the processes
        torchDist.barrier()

        # Train
        train_loss, train_acc = distTrainOneEpoch(
            model, train_dataloader, criterion, optimizer, rank, world_size
        )

        # Skip validation and epoch logging if not rank 0
        if rank != 0:
            continue

        # <Rank0> Validation
        if sys.stdout.isatty():
            val_dataloader = tqdm(val_dataloader, desc=f"Validation")
        val_loss, val_acc = validate(model, val_dataloader, criterion, rank)

        # <Rank0> Log the results of the current epoch
        logger.info(
            f"Epoch {epoch + 1}/{n_epochs}: "
            f"Train Loss: {train_loss:.4f} | "
            f"Trian Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # <Rank0> Update the best model
        if val_acc > best["val_acc"]:
            best["val_acc"] = val_acc
            best["epoch"] = epoch + 1
            best["ckpt"] = model.module.state_dict()

        # <Rank0> Save the model checkpoint
        if (epoch + 1) % ckpt_save_interval == 0:
            # Here we suppose `ckpt_save_dir` is valid
            torch.save(
                model.module.state_dict(),
                osp.join(ckpt_save_dir, f"e_{epoch + 1}.pth"),
            )

        # <Rank0> Log the statistics
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
    # <<<<<<<<<< End training <<<<<<<<<<

    # Only rank 0 will save the statistics and the best model
    if rank != 0:
        return

    # Save the statistics
    np.save(osp.join(statistics_save_dir, "train_loss.npy"), np.array(train_loss_list))
    np.save(osp.join(statistics_save_dir, "train_acc.npy"), np.array(train_acc_list))
    np.save(osp.join(statistics_save_dir, "val_loss.npy"), np.array(val_loss_list))
    np.save(osp.join(statistics_save_dir, "val_acc.npy"), np.array(val_acc_list))
    logger.info(f"Statistics saved in {statistics_save_dir}.")

    # Save the best model
    torch.save(
        best["ckpt"],
        osp.join(
            ckpt_save_dir, f"val_best_e{best['epoch']}_acc{best['val_acc']:.4f}.pth"
        ),
    )
    logger.info(
        f"The best model in validation is saved in {ckpt_save_dir}/"
        f"val_best_e{best['epoch']}_acc{best['val_acc']:.4f}.pth."
    )
