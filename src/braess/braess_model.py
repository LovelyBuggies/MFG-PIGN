import torch
from torch import nn
import numpy as np
from src.utils import MLP


class PIGN_rho(nn.Module):
    def __init__(
        self,
        f_channel_args,
        f_channel_kwargs,
        f_sum_args,
        f_sum_kwargs,
        f_x_args,
        f_x_kwargs,
    ):
        super().__init__()
        self.f_channel = MLP(f_channel_args, f_channel_kwargs)
        f_sum_args = (f_sum_args[0] + 1, f_sum_args[1], f_sum_args[2], f_sum_args[3])
        self.f_sum = MLP(f_sum_args, f_sum_kwargs)
        self.f_x = MLP(f_x_args, f_x_kwargs)

    def forward(self, x, messages=None):
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        rhos = torch.from_numpy(x).to(self.f_x.device)
        x_channel = torch.from_numpy(messages).to(
            self.f_channel.device
        )  # x_channel: (n_samples, N_edges, T, N + 1, N + 1, n_features)
        x_sum = self.f_channel(x_channel)[
            :, :, :, :, :, 0
        ]  # x_sum: (n_samples, N_edges, T, N, N)
        message_j = self.f_sum(x_sum)
        x_x = torch.concat(
            (rhos[:, :, :, :, None], message_j), dim=-1
        )  # x_x: (n_samples, N_edges, T, N, 2)
        outs = self.f_x(x_x)[:, :, :, :, 0]
        return outs


class PIGN_V(nn.Module):
    def __init__(
        self,
        f_channel_args,
        f_channel_kwargs,
        f_sum_args,
        f_sum_kwargs,
        f_x_args,
        f_x_kwargs,
    ):
        super().__init__()
        self.f_channel = MLP(f_channel_args, f_channel_kwargs)
        f_sum_args = (f_sum_args[0] + 2, f_sum_args[1], f_sum_args[2], f_sum_args[3])
        self.f_sum = MLP(f_sum_args, f_sum_kwargs)
        self.f_x = MLP(f_x_args, f_x_kwargs)

    def forward(self, x, messages=None):
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        Vs = torch.from_numpy(x).to(self.f_x.device)
        x_channel = torch.from_numpy(messages).to(
            self.f_channel.device
        )  # x_channel: (n_samples, N_edges, T + 1, N + 2, N + 2, n_features)
        x_sum = self.f_channel(x_channel)[
            :, :, :, :, :, 0
        ]  # x_sum: (n_samples, N_edges, T + 1, N + 2, N + 2)
        message_j = self.f_sum(x_sum)
        x_x = torch.concat(
            (Vs[:, :, :, :, None], message_j), dim=-1
        )  # x_x: (n_samples, N_edges, T + 1, N + 2, 2)
        outs = self.f_x(x_x)[:, :, :, :, 0]
        return outs
