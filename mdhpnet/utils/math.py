from typing import Tuple

import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def setRandomSeed(seed: int, rank: int = 0, force_deterministic: bool = False) -> None:
    """
    Set the random seed for torch and numpy.
    """
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    if force_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_cdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Cumulative Distribution Function of the given data.

    Parameters
    ----------
    data : np.ndarray
        The data to calculate CDF.
        Shape: (n,).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        [x, y], where x is the sorted data and y is the CDF.
        Shape x: (n,); Shape y: (n,).
    """
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    if np.isnan(y).any():
        raise ValueError("NaN values in CDF.")
    if np.isinf(y).any():
        raise ValueError("Inf values in CDF.")
    return x, y


def compute_cdf_for_each_dim(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the CDF for each dimension of the given data.
    Note that each dimension is calculated independently.

    Parameters
    ----------
    data : np.ndarray
        The data to calculate CDF.
        Shape: (n_samples, mdhp_dim).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (cdfs_x, cdfs_y), where cdfs_x is the sorted data and cdfs_y is the CDF.
        Shape cdfs_x: (mdhp_dim, n_samples); Shape cdfs_y: (mdhp_dim, n_samples).
    """
    cdfs_x = []
    cdfs_y = []
    # Suppose the data is in the shape of (n_samples, mdhp_dim)
    assert len(data.shape) == 2
    mdhp_dim = data.shape[1]
    for i in range(mdhp_dim):
        x, y = compute_cdf(data[:, i])
        cdfs_x.append(x)
        cdfs_y.append(y)
    # Convert to np.ndarray
    cdfs_x = np.array(cdfs_x)
    cdfs_y = np.array(cdfs_y)
    return cdfs_x, cdfs_y
