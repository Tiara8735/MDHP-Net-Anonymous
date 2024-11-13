import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import sys
from io import StringIO
from torchinfo import summary
import time

from mdhpnet.utils.math import accuracy_score, f1_score, precision_score, recall_score


@torch.inference_mode()
def testPipeline(
    device: str | torch.device,
    test_dataloader: DataLoader,
    model: Module,
    statistics_save_dir: str = None,
    logger_name: str = "TEST",
):
    """
    Test pipeline for MDHP-Net and SISSA models. Labels are required.

    Parameters
    ----------
    device : str | torch.device
        Device to use.
    test_dataloader: DataLoader
        Test dataloader.
    model : Module
        Model to train.
    statistics_save_dir : str, optional
        Directory to save the statistics (loss & acc), by default None.
    loggerName : str, optional
        Logger name, by default "TRAIN".
        You should write the logger configuration in "./configs/logger.yml".
    """
    logger = logging.getLogger(logger_name)

    logger.critical(
        f"Start testing model: {model.__class__.__name__} on {device} device."
    )

    model = model.to(device)

    pred_list, pred_label_list, label_list = [], [], []

    n_seq = model.n_seq  # Number of sequences per observation window
    model_name = model.__class__.__name__
    supposed_save_dir = osp.join(".", "results", "test", f"{model_name}_N{n_seq}")
    # Set `statistics_save_dir` if it is None
    if statistics_save_dir is None:
        statistics_save_dir = osp.join(supposed_save_dir, "statistics")
    os.makedirs(statistics_save_dir, exist_ok=True)

    # ------ Test ------
    model.eval()
    total_correct = 0
    total_samples = 0
    elapsed_time = 0
    n_samples = 0  # Number of observation windows

    # Warm up
    for i, (data, labels) in enumerate(test_dataloader):
        # [TODO][Enhancement] @tiara8735
        # |- Add an option to control the number of warm-up iterations.
        if i == 20:
            break
        data = model.preprocess(data, device)

        if i == 0:
            str_buf = StringIO()
            sys.stdout = str_buf
            summary(
                model,
                input_data={"data": data},
                col_names=("input_size", "output_size", "num_params"),
                col_width=16,
                depth=4,
                verbose=2,
            )
            sys.stdout = sys.__stdout__
            logger.info(str_buf.getvalue())

        preds = model(data)  # (batch_size, n_classes)

    if sys.stdout.isatty():
        test_dataloader = tqdm(test_dataloader, desc=f"Testing")

    # Iterate over the training dataloader
    for i, (data, labels) in enumerate(test_dataloader):
        if isinstance(data, (list, tuple)):
            n_samples += data[0].size(0)
        else:
            n_samples += data.size(0)

        start = time.time()
        data = model.preprocess(data, device)
        torch.cuda.synchronize()
        elapsed_time += time.time() - start

        labels = labels.long().to(device)
        label_list.append(labels.detach().cpu().numpy())

        start = time.time()
        preds = model(data)  # (batch_size, n_classes)
        torch.cuda.synchronize()
        elapsed_time += time.time() - start

        pred_list.append(preds.detach().cpu().numpy())

        pred_labels = preds.argmax(dim=1)  # (batch_size,)
        pred_label_list.append(pred_labels.detach().cpu().numpy())

        total_correct += (pred_labels == labels).sum().item()
        total_samples += labels.size(0)

    # Convert list to numpy array
    preds = np.concatenate(pred_list, axis=0)  # (n_samples, n_classes)
    pred_labels = np.concatenate(pred_label_list, axis=0)  # (n_samples)
    labels = np.concatenate(label_list, axis=0)  # (n_samples)

    if statistics_save_dir is not None:
        np.save(osp.join(statistics_save_dir, "preds.npy"), np.array(preds))
        np.save(osp.join(statistics_save_dir, "pred_labels.npy"), np.array(pred_labels))
        np.save(osp.join(statistics_save_dir, "labels.npy"), np.array(labels))
        logger.info(f"Statistics are saved to {statistics_save_dir}")

    acc = accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    window_speed = elapsed_time / n_samples  # seconds per observation window
    thoughput = n_seq * n_samples / elapsed_time  # message per second

    logger.info(
        f"Test Acc: {acc:.4f} | "
        f"Test F1: {f1:.4f} | "
        f"Test Precision: {precision:.4f} | "
        f"Test Recall: {recall:.4f} | "
        f"Window Speed: {window_speed:.4f} s/window | "
        f"Throughput: {thoughput:.4f} msg/s"
    )

    logger.critical(
        f"Finish testing model: {model.__class__.__name__} on {device} device."
    )

    return acc, f1, precision, recall
