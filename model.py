import numpy as np
import torch
from torch import nn


def instantiate_activation_function(function_name):
    function_dict = {
        "leaky_relu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "relu": nn.ReLU(),
        "none": None,
    }
    return function_dict[function_name]


def get_fully_connected_layer(
    input_dim,
    output_dim,
    n_hidden,
    hidden_dim,
    activation_type="leaky_relu",
    last_activation_type="tanh",
    device=None,
):
    modules = [
        nn.Linear(input_dim, hidden_dim, device=device),
    ]
    activation = instantiate_activation_function(activation_type)
    if activation is not None:
        modules.append(activation)

    if n_hidden > 1:
        for l in range(n_hidden - 1):
            modules.append(nn.Linear(hidden_dim, hidden_dim, device=device))
            activation = instantiate_activation_function(activation_type)
            if activation is not None:
                modules.append(activation)

    modules.append(nn.Linear(hidden_dim, output_dim, device=device))
    last_activation = instantiate_activation_function(last_activation_type)
    if last_activation_type == "none":
        pass
    else:
        modules.append(last_activation)

    return nn.Sequential(*modules)


class MLP(nn.Module):
    def __init__(self, nn_args, nn_kwargs):
        super().__init__()
        self.model = get_fully_connected_layer(*nn_args, **nn_kwargs)
        self.device = nn_kwargs["device"]

    def forward(self, x):
        return self.model(x)


class PIGN(nn.Module):
    def __init__(
        self, f_m_args, f_m_kwargs, f_j_args, f_j_kwargs, f_x_args, f_x_kwargs, A
    ):
        super().__init__()
        self.f_m = MLP(f_m_args, f_m_kwargs)
        self.f_j = MLP(f_j_args, f_j_kwargs)
        self.f_x = MLP(f_x_args, f_x_kwargs)
        self.A = A

    def forward(self, x, message=None):
        rho = torch.from_numpy(x).to(self.f_x.device)
        x_m = torch.from_numpy(message).to(
            self.f_m.device
        )  # x_m: (T, N, N, n_features)
        A = torch.from_numpy(self.A).to(self.f_j.device)
        x_j = self.f_m(x_m)[:, :, :, 0] * A[None, :, :]  # x_j: (T, N, N)
        message_j = self.f_j(x_j)
        x_x = torch.concat((rho[:, :, None], message_j), dim=-1)  # x_x: (T, N, 2)
        pred = self.f_x(x_x)[:, :, 0]
        return pred
