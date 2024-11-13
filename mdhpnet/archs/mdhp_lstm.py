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
import torch.nn as nn
from typing import List, Tuple


class MDHP_LSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mdhp_dim: int,
    ):
        super(MDHP_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mdhp_dim = mdhp_dim

        # Multi-Dimensional-Hawkes-Process Gate
        self.A_mdhp = nn.Parameter(torch.Tensor(mdhp_dim**2, hidden_dim))
        self.B_mdhp = nn.Parameter(torch.Tensor(mdhp_dim**2, hidden_dim))
        self.C_mdhp = nn.Parameter(torch.Tensor(mdhp_dim, hidden_dim))

        # Input Gate
        self.W_i = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_i = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.Tensor(hidden_dim))

        # Forget Gate
        self.W_f = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_f = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(hidden_dim))

        # Cell Gate
        self.W_c = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_c = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_c = nn.Parameter(torch.Tensor(hidden_dim))

        # Output Gate
        self.W_o = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_o = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.Tensor(hidden_dim))

        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            if param.data.ndimension() >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.zeros_(param.data)

    def forward(
        self,
        x: torch.Tensor,
        init_states: Tuple[torch.Tensor],
        mdhp_states: Tuple[torch.Tensor],
    ):
        # x: (seq_len, batch_size, input_dim)
        # h_t: (batch_size, hidden_dim)
        # c_t: (batch_size, hidden_dim)
        h_t, c_t = init_states
        # alpha: (batch_size, mdhp_dim ** 2)
        # beta: (batch_size, mdhp_dim ** 2)
        # theta: (batch_size, mdhp_dim)
        # tspan: (batch_size, 1)
        alpha, beta, theta, tspan = mdhp_states
        tspan = tspan.unsqueeze(1)
        outputs = []
        seq_len = x.size(0)
        for t in range(seq_len):
            # x_t: (batch_size, input_dim)
            x_t = x[t, :, :]
            # mdhp_t: (batch_size, hidden_dim)
            # mdhp_t = torch.tanh(
            #     alpha @ self.A_mdhp
            #     - torch.exp((beta * tspan) @ self.B_mdhp)
            #     + theta @ self.C_mdhp
            # )
            mdhp_t = torch.tanh(
                alpha @ self.A_mdhp - (beta * tspan) @ self.B_mdhp + theta @ self.C_mdhp
            )
            # i_t shape: (batch_size, hidden_dim)
            i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
            o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)
            c_hat_t = torch.tanh(x_t @ self.W_c + h_t @ self.U_c + self.b_c)
            c_t = mdhp_t * (f_t * c_t + i_t * c_hat_t)
            h_t = o_t * torch.tanh(c_t)
            outputs.append(h_t.unsqueeze(0))
        # Shape of outputs: (seq_len, batch_size, hidden_dim)
        outputs = torch.cat(outputs, dim=0)

        return outputs, (h_t, c_t)


if __name__ == "__main__":
    input_dim = 128
    hidden_dim = 128
    seq_len = 128
    batch_size = 128
    mdhp_dim = 16
    device = "cuda"

    lstm = MDHP_LSTM(input_dim, hidden_dim, mdhp_dim)
    # lstm = torch.jit.script(lstm)
    lstm = lstm.to(device)

    x = torch.ones(seq_len, batch_size, input_dim, device=device)
    h_0 = torch.zeros(batch_size, hidden_dim, device=device)
    c_0 = torch.zeros(batch_size, hidden_dim, device=device)
    init_states = (h_0, c_0)
    alpha = torch.ones(batch_size, mdhp_dim**2, device=device)
    beta = torch.ones(batch_size, mdhp_dim**2, device=device)
    theta = torch.ones(batch_size, mdhp_dim, device=device)
    tspan = torch.ones(batch_size, 1, device=device)
    mdhp_states = [alpha, beta, theta, tspan]

    outputs, (h_t, c_t) = lstm(x, init_states, mdhp_states)
    print(outputs.shape)  # torch.Size([5, 3, 20])
    print(h_t.shape)  # torch.Size([3, 20])
    print(c_t.shape)  # torch.Size([3, 20])
