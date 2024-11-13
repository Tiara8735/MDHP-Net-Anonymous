import numpy as np
from typing import List, Tuple
import os
import os.path as osp

from mdhpnet.models import MDHP_GDS


def solve_mdhp_params(
    timestamps: List[List[float]],
    device: str = "cuda",
    loss_list_save_path: str = None,
    alpha_save_path: str = None,
    beta_save_path: str = None,
    theta_save_path: str = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the parameters of Multi-Dimensional Hawkes Process using Gradient Descent.

    Parameters
    ----------
    timestamps : List[List[float]]
        The time of occurance of each dimension.
    device : str, optional
        Host device for solving the parameters, by default "cuda".
    loss_list_save_path : str, optional
        Path to save the loss list during calculation, by default None.
    alpha_save_path : str, optional
        Path the save alpha, by default None.
    beta_save_path : str, optional
        Path to save beta, by default None.
    theta_save_path : str, optional
        Path to save theta, by default None.

    Kwargs
    ------
    max_epochs : int, optional
        Maximum epochs for fitting, by default 10000.
    tolerance : float, optional
        Tolerance for convergence, by default 1e-4.
    check_interval : int, optional
        Interval for checking convergence, by default 50.
    use_torch_compile : bool, optional
        Whether to use torch.jit.script to compile the model, by default True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        [Alpha, Beta, Theta], with shape [(Dim, Dim), (Dim, Dim), (Dim,)].
    """
    solver = MDHP_GDS(timestamps=timestamps, logger_name="MDHP_GDS")
    solver.to(device)
    loss_list = solver.fit(**kwargs)
    alpha, beta, theta = solver.get_parameters()

    if loss_list_save_path and loss_list_save_path.endswith(".npy"):
        os.makedirs(osp.dirname(loss_list_save_path), exist_ok=True)
        np.save(loss_list_save_path, np.array(loss_list, dtype=np.float32))

    if alpha_save_path and alpha_save_path.endswith(".npy"):
        os.makedirs(osp.dirname(alpha_save_path), exist_ok=True)
        np.save(alpha_save_path, alpha)

    if beta_save_path and beta_save_path.endswith(".npy"):
        os.makedirs(osp.dirname(beta_save_path), exist_ok=True)
        np.save(beta_save_path, beta)

    if theta_save_path and theta_save_path.endswith(".npy"):
        os.makedirs(osp.dirname(theta_save_path), exist_ok=True)
        np.save(theta_save_path, theta)

    # Release memory of the model
    del solver

    return alpha, beta, theta


def gen_example_timestamps(
    mdhp_dim: int, max_len: int, min_len: int
) -> List[List[float]]:
    """
    A simple function to generate example timestamps for testing.

    Parameters
    ----------
    mdhp_dim : int
        Dimension of the Multi-Dimensional Hawkes Process.
    max_len : int
        Max length of single dimension.
    min_len : int
        Min length of single dimension.

    Returns
    -------
    List[List[float]]
        List of timestamps for each dimension.
    """
    timestamps = [[] for _ in range(mdhp_dim)]
    for i in range(mdhp_dim):
        timestamps[i] = np.sort(
            np.random.uniform(0, 10, size=np.random.randint(min_len, max_len))
        )
    return timestamps
