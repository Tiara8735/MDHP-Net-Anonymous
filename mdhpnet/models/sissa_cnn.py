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

from mdhpnet.archs import MLP, SMB, RSAB, ActConv


class SISSA_CNN(nn.Module):
    def __init__(
        self,
        n_seq: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        # [TODO][BUG] @tiara8735
        # |- Error when compiling SISSA_CNN.
        # |- But I think this model has already been fast enough without compiling? 🤔
        disable_torch_compile: bool = True,
    ):
        super(SISSA_CNN, self).__init__()

        self.n_seq = n_seq
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.disable_torch_compile = disable_torch_compile

        self.smb = SMB(
            n_seq=n_seq,
            in_dim=input_dim,
            out_dim=hidden_dim,
            n_channels=1,
            mode="BNL",
            non_linear_layer=nn.ReLU(),
        )
        self.conv1 = ActConv(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            width=n_seq,
            height=hidden_dim,
        )
        self.conv2 = ActConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            width=self.conv1.output_width,
            height=self.conv1.output_height,
        )

        self.channel_layer = MLP(
            n_in=hidden_dim,
            n_hidden=hidden_dim,
            n_out=1,
        )
        self.rsab = RSAB(
            seq_len=self.conv2.output_width,
            in_dim=self.conv2.output_height,
            out_dim=self.conv2.output_height,
        )
        self.mlp = MLP(
            n_in=self.conv2.output_width * self.conv2.output_height,
            n_hidden=hidden_dim,
            n_out=output_dim,
        )

        self.forward = torch.compile(self.forward, disable=disable_torch_compile)

    def preprocess(self, data: torch.Tensor, device: str | int) -> Tuple[torch.Tensor]:
        obs_window = data.to(device)
        return obs_window

    def forward(self, data: Tuple[torch.Tensor]) -> torch.Tensor:
        x = data
        # x: (batch_size, seq_len, input_dim)
        x = self.smb(x)
        # x: (batch_size, seq_len, hidden_dim)
        x = x.unsqueeze(1)
        # x: (batch_size, 1, seq_len, hidden_dim)
        x = self.conv1(x)
        x = self.conv2(x)
        # x: (batch_size, hidden_dim, w, h)
        x = x.permute(0, 2, 3, 1)  # Move channel to the last dimension
        # x: (batch_size, w, h, hidden_dim)
        x = self.channel_layer(x).squeeze(-1)
        # x: (batch_size, w, h)
        x = self.rsab(x)
        # x: (batch_size, w, h)
        x = x.reshape(-1, x.size(-2) * x.size(-1))
        # x: (batch_size, w * h)
        x = self.mlp(x)
        # x: (batch_size, output_dim)
        return x
