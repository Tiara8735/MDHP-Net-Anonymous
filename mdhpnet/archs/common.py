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
from torch.nn import functional as F
from typing import Any


class SMB(nn.Module):
    def __init__(
        self,
        n_seq: int,
        in_dim: int,
        out_dim: int,
        n_channels: int = None,
        mode: str = "NBL",
        non_linear_layer: Any = lambda x: x,
    ):
        """
        Squence Mapping Block

        Parameters
        ----------
        n_seq: int
            Number of the sequences in the input tensor.
        in_dim: int
            Dimension of the input tensor, i.e., the length of a feature vector.
        out_dim : int
            Dimension of the output tensor.
        n_channels : int, optional
            Number of channels in the input tensor, by default None.
        mode : str, optional
            Mode of the block, by default "NBL". Options:
            - NBL: (n_seq, batch_size, in_dim) -> (batch_size, n_seq, out_dim)
            - BNL: (batch_size, n_seq, in_dim) -> (batch_size, n_seq, out_dim)
            - BCNL: (batch_size, n_channels, n_seq, in_dim) -> (batch_size, n_channels, n_seq, out_dim)
        non_linear_layer : Any, optional
            Optional Non-Linear layer, by default `lambdax:x`.
        """
        super(SMB, self).__init__()
        self.n_seq = n_seq
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.non_linear_layer = non_linear_layer
        if mode == "NBL":
            self.preTrans = lambda x: x.transpose(0, 1).reshape(-1, self.in_dim)
            self.postTrans = lambda x: x.reshape(
                -1, self.n_seq, self.out_dim
            ).transpose(0, 1)
        elif mode == "BNL":
            self.preTrans = lambda x: x.reshape(-1, self.in_dim)
            self.postTrans = lambda x: x.reshape(-1, self.n_seq, self.out_dim)
        elif mode == "BCNL" and n_channels is not None:
            self.n_channles = n_channels
            self.preTrans = lambda x: x.reshape(-1, self.in_dim)
            self.postTrans = lambda x: x.reshape(
                -1, self.n_channles, self.n_seq, self.out_dim
            )
        else:
            raise ValueError("Invalid mode")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preTrans(x)
        x = self.linear(x)
        x = self.non_linear_layer(x)
        x = self.postTrans(x)
        return x


class CMB(nn.Module):
    def __init__(
        self,
        n_channels: int,
        img_width: int,
        img_height: int,
        n_out: int,
    ):
        """
        Channel Mapping Block

        Breif
        -----
        This block is used to map the channels of the input image to the
        channels of the output image (by each pixel).
        """
        super(CMB, self).__init__()
        self.n_channels = n_channels
        self.n_out = n_out
        self.linear = nn.Linear(n_channels, n_out)
        self.img_width = img_width
        self.img_height = img_height

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (..., n_channels, img_width, img_height)
        x = x.view(-1, self.n_channels)  # (..., n_channels)
        x = self.linear(x)  # (..., n_out)
        x = x.view(
            -1, self.img_width, self.img_height, self.n_out
        )  # (..., img_width, img_height, n_out)
        # Output shape: (..., img_width, img_height, n_out)
        return x


class RSAB(nn.Module):
    def __init__(self, seq_len: int, in_dim: int, out_dim: int):
        """
        Residual Attention Block
        """
        super(RSAB, self).__init__()
        if in_dim != out_dim:
            raise ValueError(
                "`n_in` must be equal to `n_out` in Residual Attention Block (RAB)."
            )
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, n_seq, n_in)
        residual = x
        x = x.reshape(-1, self.in_dim)  # (batch_size * n_seq, n_in)
        q = self.W_q(x)  # (batch_size * n_seq, n_out)
        k = self.W_k(x)  # (batch_size * n_seq, n_out)
        v = self.W_v(x)  # (batch_size * n_seq, n_out)
        q = q.view(-1, self.seq_len, self.out_dim)  # (batch_size, n_seq, n_out)
        k = k.view(-1, self.seq_len, self.out_dim)  # (batch_size, n_seq, n_out)
        v = v.view(-1, self.seq_len, self.out_dim)  # (batch_size, n_seq, n_out)
        # Scaled dot-product attention
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.out_dim, dtype=torch.float)
        )  # (batch_size, n_seq, n_seq)
        attn_score = torch.matmul(
            F.softmax(attn_score, dim=-1), v
        )  # (batch_size, n_seq, n_out)
        # Output shape: (batch_size, n_seq, n_out)
        return attn_score + residual


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_hidden: int):
        """
        Multi-Layer Perceptron
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, n_in)
        x = F.relu(self.fc1(x))  # (batch_size, n_hidden)
        x = self.fc2(x)  # (batch_size, n_out)
        # Output shape: (batch_size, n_out)
        return x


class ActConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        width: int,
        height: int,
    ):
        """
        Activated Convolutional Layer (with MaxPooling and Dropout)
        """
        super(ActConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.output_width = self.sizeAfterConv(width, kernel_size, stride, padding) // 2
        self.output_height = (
            self.sizeAfterConv(height, kernel_size, stride, padding) // 2
        )

    @staticmethod
    def sizeAfterConv(size: int, kernel_size: int, stride: int, padding: int) -> int:
        return (size - kernel_size + 2 * padding) // stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
