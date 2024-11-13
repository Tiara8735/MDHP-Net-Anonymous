import numpy as np
import os
import os.path as osp
import logging
from tqdm import tqdm

from mdhpnet.utils import solve_mdhp_params, initLoggers


initLoggers(r"configs/loggers.yml")
basic_logger = logging.getLogger("BASIC")

# fmt: off
all_raw_data_paths = (
    "data/raw/train_5_5/data.npy",
    "data/raw/val_5_5/data.npy",
    "data/raw/train_4_4/data.npy",
    "data/raw/val_4_4/data.npy"
)
all_target_data_dirs = (
    "data/rate_0.5_victim_5/train",
    "data/rate_0.5_victim_5/val",
    "data/rate_0.4_victim_4/train",
    "data/rate_0.4_victim_4/val",
)

unique_src_ips = [
    10001, 10002, 10003, 10004, 10006, 10007, 10008, 10101,
    10102, 10103, 10104, 10105, 10106, 10107, 10108, 10109,
]

# Create a mapping from IP to index
ip_to_index = {ip: i for i, ip in enumerate(unique_src_ips)}


def get_timestamps(time, ids):
    timestemps = [[] for _ in range(len(unique_src_ips))]
    for t, i in zip(time, ids):
        timestemps[ip_to_index[i]].append(t)
    return timestemps


for raw_data_path, target_data_dir in zip(all_raw_data_paths, all_target_data_dirs):
    os.makedirs(target_data_dir, exist_ok=True)   

    basic_logger.info(f"Read data from {raw_data_path}")
    raw_data = np.load(raw_data_path)  # Shape: (30748, 128, 26)
    basic_logger.info(f"Data shape: {raw_data.shape}")

    alphas, betas, thetas = [], [], []

    # 30748 iterations
    for idx in tqdm(range(raw_data.shape[0]), desc="Processing data"):
        msg_window = raw_data[idx]  # (128, 26)
        time = msg_window[:, 0] * 10
        ids = msg_window[:, -1]
        mapped_timestamps = get_timestamps(time, ids)
        alpha, beta, theta = solve_mdhp_params(
            timestamps=mapped_timestamps,
            device="cuda",
            max_epochs=500,
            check_interval=25,
        )
        alphas.append(alpha)  # (16, 16)
        betas.append(beta)  # (16, 16)
        thetas.append(theta)  # (16,)

    alphas = np.array(alphas)  # (30748, 16, 16)
    betas = np.array(betas)  # (30748, 16, 16)
    thetas = np.array(thetas)  # (30748, 16)

    np.save(osp.join(target_data_dir, "data.npy"), raw_data[:, :, 0:-1])  # (30748, 128, 25)
    np.save(osp.join(target_data_dir, "alpha.npy"), alphas)  # (30748, 16, 16)
    np.save(osp.join(target_data_dir, "beta.npy"), betas)  # (30748, 16, 16)
    np.save(osp.join(target_data_dir, "theta.npy"), thetas)  # (30748, 16)

    basic_logger.info(f"Finished processing data: {raw_data_path}")
