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
from torch.optim import AdamW, Adam

import numpy as np
from typing import List, Tuple
import logging


class MDHP_GDS:
    """
    Gradient Descent Solver for Multi-Dimensional Hawkes Process.
    """

    def __init__(
        self,
        timestamps: List[np.ndarray],
        learning_rate=0.01,
        weight_decay=0.01,
        logger_name: str = "MDHP_GDS",
    ):
        """
        Calculate `Alpah`, `Beta`, `Theta` of MDHP with maximum likelihood estimation.

        Parameters
        ----------
        timestamps : List[np.ndarray]
            The timestamps of occurance of a event of each dimension.
            `time[i][k]`: Time of the k-th event of the i-th dimension.
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.logger = logging.getLogger(name=logger_name)

        # Dimension of the Hawkes process
        self.dim = len(timestamps)

        # [TODO|@tiara8735] Use registry to provide different optimizers.
        self.optimizer = None

        # Init parameters
        self.init_parameters()

        # Time of occurance of a event of each dimension
        self.time = self._standardize_time(timestamps)
        # Span of the time series
        self.span = max([t[-1] for t in self.time if len(t) > 0])

        # The max length of the time series
        self.maxTimeLen = max([len(t) for t in timestamps])
        # Padding the time series to the same length with -inf; Shape: (Dim, MaxLen)
        self.paddedTime = self._padTime(self.maxTimeLen, -1)
        # Shape: (Dim, maxTimeLen, Dim, maxTimeLen)
        self.tMpT, self.tMpT_Mask = self._t_minus_paddedTime()

    def fit(
        self,
        max_epochs: int = 1000,
        tolerance: float = 1e-4,
        check_interval: int = 50,
        use_torch_compile: bool = True,
        optimize_likelihood: bool = True,
    ) -> List:
        """
        Fit the model with maximum likelihood estimation.

        Parameters
        ----------
        """

        loss_list = []
        avg_loss_change = float("inf")

        if not optimize_likelihood:
            self.logger.warning(
                "The model is running without optimization. "
                "This is only for testing purpose."
            )
            self._ln_likelihood = self._ln_likelihood_no_opt

        if use_torch_compile:
            self._ln_likelihood = torch.compile(self._ln_likelihood)

        for i in range(max_epochs):
            self.optimizer.zero_grad()
            loss = -self._ln_likelihood()

            self.logger.debug(f"Epoch {i+1:4}/{max_epochs:4}, Loss: {loss.item()}")
            loss_list.append(loss.item())

            if i % check_interval == 0 and i > 0:
                recent_losses = loss_list[-check_interval:]
                avg_loss_change = np.mean(np.diff(recent_losses))
                # Stop training if loss is rising or loss change is below tolerance
                if avg_loss_change > 0:
                    self.logger.critical(
                        f"Model has diverged at epoch {i}. "
                        f"Average loss change {avg_loss_change:.6f} "
                        f"is above zero."
                    )
                    break
                elif -avg_loss_change < tolerance:
                    self.logger.critical(
                        f"Model has converged at epoch {i}. "
                        f"Average loss change {avg_loss_change:.6f} "
                        f"is below tolerance {tolerance}."
                    )
                    break

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                parameters=[self.alpha, self.beta, self.theta], max_norm=1.0
            )

            self.optimizer.step()
            self._to_possitive()

        return loss_list

    def init_parameters(self):
        """
        Initialize the parameters of the model.
        """
        self.alpha = nn.Parameter(torch.ones((self.dim, self.dim), dtype=torch.float32))
        self.beta = nn.Parameter(torch.ones((self.dim, self.dim), dtype=torch.float32))
        self.theta = nn.Parameter(torch.ones((self.dim), dtype=torch.float32))
        self.optimizer = Adam(
            [self.alpha, self.beta, self.theta],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get the parameters of the model.
        """
        return (
            self.alpha.detach().cpu().numpy(),
            self.beta.detach().cpu().numpy(),
            self.theta.detach().cpu().numpy(),
        )

    def to(self, device: str):
        """
        Move the model to a specific device.
        """
        self.alpha = nn.Parameter(self.alpha.to(device))
        self.beta = nn.Parameter(self.beta.to(device))
        self.theta = nn.Parameter(self.theta.to(device))
        self.paddedTime = self.paddedTime.to(device)
        self.tMpT = self.tMpT.to(device)
        self.tMpT_Mask = self.tMpT_Mask.to(device)
        self.optimizer = Adam(
            [self.alpha, self.beta, self.theta],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return self

    def _ln_likelihood(self) -> torch.Tensor:
        return self._Part1() + self._Part2() + self._Part3()

    def _to_possitive(self):
        with torch.no_grad():
            self.alpha[self.alpha <= 0] = 1e-9
            self.beta[self.beta <= 0] = 1e-9
            self.theta[self.theta <= 0] = 1e-9

    @staticmethod
    def _min_max_transformer(tVec, oldMin, oldMax, newMin, newMax):
        tVec = np.array(tVec, dtype=np.float32)
        return (tVec - oldMin) / (oldMax - oldMin) * (newMax - newMin) + newMin

    def _standardize_time(
        self, time: List[np.ndarray] | List[List]
    ) -> List[torch.Tensor]:
        """
        Standardize the time series and convert to torch.Tensor.

        Parameters
        ----------
        time : List[np.ndarray] | List[List]
            The time series of the events.
        """
        oldMin = min([t[0] for t in time if len(t) > 0])
        oldMax = max([t[-1] for t in time if len(t) > 0])
        newMin = 0
        newMax = 80  # [Note] exp(>90) for float32 will be NaN.
        return [
            (
                torch.tensor(
                    self._min_max_transformer(t, oldMin, oldMax, newMin, newMax),
                    dtype=torch.float32,
                )
                if len(t) > 0
                else torch.tensor([], dtype=torch.float32)
            )
            for t in time
        ]

    def _padTime(self, length: int, value=0.0) -> torch.Tensor:
        """
        Pad the time series to the same length with a specific value.

        Parameters
        ----------
        length: int
            The length to pad the time series.
        value : torch.float32
            The value to pad the time series.
        """
        return torch.stack(
            [
                torch.nn.functional.pad(t, (0, length - t.size(0)), "constant", value)
                for t in self.time
            ]
        )

    def _t_minus_paddedTime(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # `time_expanded`: (dim, maxTimeLen, 1, 1)
        time_expanded = self.paddedTime[:, :, None, None]
        # `time_repeated`: (dim, maxTimeLen, dim, maxTimeLen)
        time_repeated = time_expanded.repeat(1, 1, self.dim, self.maxTimeLen)

        condition1 = time_repeated > self.paddedTime[None, None, :, :]
        condition2 = time_repeated != -1
        condition3 = self.paddedTime != -1

        return (
            time_repeated - self.paddedTime[None, None, :, :],
            condition1 & condition2 & condition3,
        )

    def _Part1(self) -> torch.Tensor:
        # `alpha_expanded`: (dim, 1, dim, 1)
        alpha_expanded = self.alpha[:, None, :, None]
        # `beta_expanded`: (dim, 1, dim, 1)
        beta_expanded = self.beta[:, None, :, None]
        # `exp_term`: (dim, maxTimeLen, dim, maxTimeLen)
        exp_term = (
            torch.exp(torch.clamp_max(-beta_expanded * self.tMpT, 80.0))
            * self.tMpT_Mask.float()
        )
        # `alpha_exp_term`: (dim, maxTimeLen, dim, maxTimeLen)
        alpha_exp_term = alpha_expanded * exp_term
        # Sum up dimension j and k;
        # `sum_jk`: (dim, maxTimeLen)
        sum_jk = torch.sum(alpha_exp_term, dim=[2, 3])
        # `theta_expanded`: (dim, 1)
        theta_expanded = self.theta[:, None]
        # `log_term`: (dim)
        log_term = torch.log(theta_expanded + sum_jk)
        # Output: (1)
        return torch.sum(log_term)

    def _Part2(self) -> torch.Tensor:
        part2 = -self.span * torch.sum(self.theta)
        return part2

    def _Part3(self) -> torch.Tensor:
        # Broadcast `paddedTime` from (dim, maxTimeLen) to (dim, dim, maxTimeLen)
        bcastedTime = self.paddedTime[:, None, :].repeat(1, self.dim, 1)
        bcastedTimeMask = (bcastedTime != -1).float()
        # `inner_k`: (dim, dim, maxTimeLen)
        inner_k = (
            (
                torch.exp(
                    torch.clamp_max(
                        -self.beta[:, :, None] * (self.span - bcastedTime), 80.0
                    )
                )
                - 1
            )
            * bcastedTimeMask
        ).transpose(0, 1)
        # `sum_inner_k`: (dim, dim)
        sum_inner_k = torch.sum(inner_k, dim=2)
        # Output: (1)
        return torch.sum(self.alpha / self.beta * sum_inner_k)

    def _ln_likelihood_no_opt(self) -> torch.Tensor:
        """
        Calculate the log likelihood of the model without optimization.
        This function should only be used for testing.
        """
        part1 = 0
        for i in range(0, self.dim):
            for t in self.time[i]:
                ln_inner = self.theta[i]
                for j in range(0, self.dim):
                    for k in range(0, len(self.time[j])):
                        if self.time[j][k] < t:
                            ln_inner = ln_inner + self.alpha[i][j] * torch.exp(
                                torch.clamp_max(
                                    -self.beta[i][j] * (t - self.time[j][k]), 80.0
                                )
                            )
                part1 = part1 + torch.log(ln_inner)
        part2 = -self.span * torch.sum(self.theta)
        part3 = 0
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                adb = self.alpha[i][j] / self.beta[i][j]
                sum_inner_k = 0
                for k in range(0, len(self.time[j])):
                    sum_inner_k = (
                        sum_inner_k
                        + torch.exp(
                            torch.clamp_max(
                                -self.beta[i][j] * (self.span - self.time[j][k]), 80.0
                            )
                        )
                        - 1
                    )
                part3 = part3 + adb * sum_inner_k
        return part1 + part2 + part3
