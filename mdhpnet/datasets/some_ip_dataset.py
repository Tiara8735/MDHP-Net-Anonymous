import torch
from torch.utils.data import Dataset
import numpy as np
import os.path as osp

class SomeIP_Dataset(Dataset):
    def __init__(self, data_dir: str, n_seq: int = None):
        """
        A Simple Implementation of SomeIP MDHP Dataset.
        You should prepare the dataset in the following format:
        - data.npy: (n_samples, n_seq, n_features)
        - labels.npy: (n_samples, )

        Parameters
        ----------
        data_dir: str
            Directory of the dataset.
        n_seq: int
            Number of the sequences in the input tensor.
        """
        obs_window_path = osp.join(data_dir, 'data.npy')
        label_path = osp.join(data_dir, 'labels.npy')

        self.label = np.load(label_path)  # (n_samples,)
        self.obs_window = np.load(obs_window_path)  # (n_samples, seq_len, n_features)
        self.n_seq = n_seq

        if self.n_seq is not None:
            self.obs_window = self.obs_window[:, :self.n_seq]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # One obs_window shape: (seq_len, n_features)
        obs_window = torch.tensor(self.obs_window[idx], dtype=torch.float32)
        # Add gaussian noise to the observation window
        obs_window += torch.randn_like(obs_window) * 0.01
        # label shape: (1,)
        label = torch.tensor(self.label[idx], dtype=torch.float32)
        return obs_window, label
