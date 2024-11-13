import torch
from torch.utils.data import Dataset
import numpy as np
import os.path as osp

class SomeIP_MDHP_Dataset(Dataset):
    def __init__(self, data_dir: str, n_seq: int = None):
        """
        A Simple Implementation of SomeIP MDHP Dataset.
        You should prepare the dataset in the following format:
        - data.npy: (n_samples, seq_len, n_features)
        - alpha.npy: (n_samples, mdhp_dim, mdhp_dim)
        - beta.npy: (n_samples, mdhp_dim, mdhp_dim)
        - theta.npy: (n_samples, mdhp_dim)
        - labels.npy: (n_samples,)

        Parameters
        ----------
        data_dir: str
            Directory of the dataset.
        n_seq: int
            Number of the sequences in the input tensor.
        """
        obs_window_path = osp.join(data_dir, 'data.npy')
        alpha_path = osp.join(data_dir, 'alpha.npy')
        beta_path = osp.join(data_dir, 'beta.npy')
        theta_path = osp.join(data_dir, 'theta.npy')
        label_path = osp.join(data_dir, 'labels.npy')

        self.label = np.load(label_path)
        self.obs_window = np.load(obs_window_path)
        self.alpha = np.load(alpha_path)
        self.beta = np.load(beta_path)
        self.theta = np.load(theta_path)

        self.n_seq = n_seq
        if self.n_seq is not None:
            self.obs_window = self.obs_window[:, :self.n_seq]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # obs_window shape: (seq_len, n_features)
        obs_window = torch.tensor(self.obs_window[idx], dtype=torch.float32)
        # Add gaussian noise to the observation window
        obs_window += torch.randn_like(obs_window) * 0.01
        # Suppose the first column of obs_window is the time
        tspan = torch.max(obs_window[:, 0]) - torch.min(obs_window[:, 0])
        assert tspan > 0, "tspan should be greater than 0"
        # alpha, beta shape: (mdhp_dim, mdhp_dim)
        alpha = torch.tensor(self.alpha[idx], dtype=torch.float32).view(-1)
        beta = torch.tensor(self.beta[idx], dtype=torch.float32).view(-1)
        theta = torch.tensor(self.theta[idx], dtype=torch.float32)
        # label shape: (1,)
        label = torch.tensor(self.label[idx], dtype=torch.float32)
        return (obs_window, alpha, beta, theta, tspan), label
