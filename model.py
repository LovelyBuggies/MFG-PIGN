import numpy as np
import torch
from torch import nn


class Multiply(nn.Module):
    def __init__(self, scale):
        super(Multiply, self).__init__()
        self.scale = scale

    def forward(self, tensors):
        return self.scale * tensors


class Normalization(nn.Module):
    def __init__(self, mean, std, device):
        super(Normalization, self).__init__()
        self.device = device
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, tensors):
        norm_tensor = (tensors - self.mean) / self.std
        return norm_tensor


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
    mean=0,
    std=1,
):
    modules = [
        Normalization(mean, std, device),
        nn.Linear(input_dim, hidden_dim, device=device),
    ]
    activation = instantiate_activation_function(activation_type)
    if activation is not None:
        modules.append(activation)

    # add hidden model
    if n_hidden > 1:
        for l in range(n_hidden - 1):
            modules.append(nn.Linear(hidden_dim, hidden_dim, device=device))
            activation = instantiate_activation_function(activation_type)
            if activation is not None:
                modules.append(activation)

    # add the last layer
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
        for l in self.model:
            x = l(x)

        return x


class MPNN(nn.Module):
    def __init__(self, f_x_args, f_x_kwargs, f_m_args, f_m_kwargs, A):
        super().__init__()
        self.f_x = MLP(f_x_args, f_x_kwargs)
        self.f_m = MLP(f_m_args, f_m_kwargs)
        self.A = A

    def forward(self, x):
        # x is of shape(batch, N, F)
        x = np.float32(x)
        A = np.float32(self.A)
        N = x.shape[1]
        rho = x[:, :, 0]
        V = x[:, :, 1]
        rho_i = np.repeat(rho[:, np.newaxis, :], N, axis=1)
        rho_j = np.repeat(rho[:, :, np.newaxis], N, axis=2)
        V_i = np.repeat(V[:, np.newaxis, :], N, axis=1)
        V_j = np.repeat(V[:, :, np.newaxis], N, axis=2)

        # convert to torch
        message = np.concatenate(
            [
                (rho_i + rho_j)[:, :, :, np.newaxis],
                (V_i + V_j)[:, :, :, np.newaxis],
            ],
            axis=-1,
        )
        torch_x = torch.from_numpy(message)
        A = torch.from_numpy(A).to(self.f_m.device)
        message_embed = self.f_m(torch_x.to(self.f_m.device))
        message_sum = torch.sum(message_embed[:, :, :, 0] * A[None, :, :], 2)
        rho = torch.Tensor(rho)
        torch_x = torch.concat((rho[:, :, None], message_sum[:, :, None]), axis=-1)
        dot_rho_i = self.f_x(torch_x)
        return dot_rho_i
