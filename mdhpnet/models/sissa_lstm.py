"""
Copyright 2023-2024 Anonymous

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
from torch import nn
from typing import Tuple

from mdhpnet.archs import MLP, SMB, RSAB


class SISSA_LSTM(nn.Module):
    def __init__(
        self,
        n_seq: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_lstm: int,
        disable_torch_compile: bool = False,
    ):
        super(SISSA_LSTM, self).__init__()

        self.n_seq = n_seq
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_lstm = n_lstm
        self.disable_torch_compile = disable_torch_compile

        self.smb = SMB(
            n_seq=n_seq,
            in_dim=input_dim,
            out_dim=hidden_dim,
            mode="NBL",
            non_linear_layer=nn.ReLU(),
        )
        self.lstm_list = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm,
            batch_first=False,  # (seq_len, batch_size, input_dim)
        )
        self.rsab = RSAB(
            seq_len=n_seq,
            in_dim=hidden_dim,
            out_dim=hidden_dim,
        )
        self.mlp = MLP(
            n_in=n_seq * hidden_dim,
            n_hidden=hidden_dim,
            n_out=output_dim,
        )

        self.forward = torch.compile(self.forward, disable=disable_torch_compile)

    def preprocess(self, data: torch.Tensor, device: str | int) -> torch.Tensor:
        # h_0 = torch.zeros(batch_size, self.hidden_dim).to(device)
        # c_0 = torch.zeros(batch_size, self.hidden_dim).to(device)
        # init_states = (h_0, c_0)
        obs_window = data.to(device)
        # Convert obs_window
        # from (batch_size, n_seq, len_seq)
        # to (n_seq, batch_size, len_seq)
        obs_window = obs_window.permute(1, 0, 2)
        return obs_window

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = data
        # x: (n_seq, batch_size, input_dim)
        x = self.smb(x)
        # x: (n_seq, batch_size, hidden_dim)
        x, _ = self.lstm_list(x)
        # x: (n_seq, batch_size, hidden_dim)
        x = x.transpose(0, 1)
        # x: (batch_size, n_seq, hidden_dim)
        x = self.rsab(x)
        # x: (batch_size, n_seq, hidden_dim)
        x = x.reshape(-1, x.size(-2) * x.size(-1))
        # x: (batch_size, n_seq * hidden_dim)
        x = self.mlp(x)
        # x: (batch_size, output_dim)
        return x
