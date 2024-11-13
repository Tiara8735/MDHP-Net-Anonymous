"""
Copyright 2024 Anonymous

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

from mdhpnet.archs import MDHP_LSTM, MLP, SMB, RSAB

# [TODO][Question] @tiara8735
# |- I'm not sure if this is the right place to put this line.
torch.set_float32_matmul_precision("high")


class MDHP_NET(nn.Module):
    def __init__(
        self,
        n_seq: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        mdhp_dim: int,
        n_mdhp_lstm: int,
        disable_torch_compile: bool = False,
    ):
        super(MDHP_NET, self).__init__()

        self.n_seq = n_seq
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mdhp_dim = mdhp_dim
        self.n_mdhp_lstm = n_mdhp_lstm
        self.disable_torch_compile = disable_torch_compile

        self.smb = SMB(
            n_seq=n_seq,
            in_dim=input_dim,
            out_dim=hidden_dim,
            mode="NBL",
            non_linear_layer=nn.ReLU(),
        )
        self.mdhp_lstm_list = nn.ModuleList(
            [
                MDHP_LSTM(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    mdhp_dim=mdhp_dim,
                )
                for _ in range(n_mdhp_lstm)
            ]
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

        self.preprocess = torch.compile(self.preprocess, disable=disable_torch_compile)
        self.forward = torch.compile(self.forward, disable=disable_torch_compile)

    def preprocess(
        self, data: Tuple[torch.Tensor], device: str | int
    ) -> Tuple[torch.Tensor]:
        batch_size = data[0].size(0)
        h_0 = torch.zeros(batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(batch_size, self.hidden_dim).to(device)
        init_states = (h_0, c_0)
        obs_window, alpha, beta, theta, tspan = [x.to(device) for x in data]
        # Convert obs_window
        # from (batch_size, n_seq, len_seq)
        # to (n_seq, batch_size, len_seq)
        obs_window = obs_window.permute(1, 0, 2)
        mdhp_params = (alpha, beta, theta, tspan)
        return obs_window, init_states, mdhp_params

    def forward(
        self,
        # data is a tuple of (x, init_states, mdhp_params)
        # |- x: torch.Tensor,
        # |- init_states: Tuple[torch.Tensor],
        # |- mdhp_params: Tuple[torch.Tensor],
        data: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        # [TODO|@tiara8735]
        # >  Maybe I should put init_states inside each MDHP-LSTM?
        x, init_states, mdhp_params = data
        # x: (n_seq, batch_size, input_dim)
        x = self.smb(x)
        # x: (n_seq, batch_size, hidden_dim)
        for mdhp_lstm in self.mdhp_lstm_list:
            x, init_states = mdhp_lstm(x, init_states, mdhp_params)
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
