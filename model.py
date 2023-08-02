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
        # Normalization(mean, std, device),
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
    def __init__(
        self, f_x_args, f_x_kwargs, f_m_args, f_m_kwargs, f_new_args, f_new_kwargs, A
    ):
        super().__init__()
        self.f_x = MLP(f_x_args, f_x_kwargs)
        self.f_m = MLP(f_m_args, f_m_kwargs)
        self.f_new = MLP(f_new_args, f_new_kwargs)
        self.A = A

    def forward(self, x, cumulative_transitions):
        A = np.float32(self.A)
        N, T = x.shape
        rho = torch.from_numpy(x).to(self.f_x.device).float()
        rho_i = np.repeat(rho[:, None, :], N, axis=1)
        rho_j = np.repeat(rho[:, :, None], N, axis=2)
        rho_ij = (rho_i + rho_j) / 2

        # message
        message = np.zeros((T, N, N), dtype=np.float32)
        for t in range(T):
            message[t, :, :] = cumulative_transitions[t]

        message = torch.concat(
            (
                torch.from_numpy(message[:, :, :, None]),
                rho_i[:, :, :, None],
                rho_j[:, :, :, None],
            ),
            dim=-1,
        )
        x_m = message.to(self.f_m.device)
        A = torch.from_numpy(A).to(self.f_m.device)
        x_new = self.f_m(x_m)[:, :, :, 0] * A[None, :, :]
        message_new = self.f_new(x_new)
        x_x = torch.concat((rho[:, :, None], message_new), dim=-1)
        pred = self.f_x(x_x)[:, :, 0]
        return pred

    def transition_loss(self, pred, transitions, init_rho, loss_func):
        N, T = pred.shape
        loss_ic = loss_func(
            torch.matmul(torch.from_numpy(transitions[0]).float(), pred[:, 0]),
            torch.from_numpy(init_rho),
        )
        loss_physics = 0.0
        for t in range(1, T):
            loss_physics += loss_func(
                torch.matmul(torch.from_numpy(transitions[t]).float(), pred[:, t - 1]),
                pred[:, t],
            )

        return loss_ic * 0.6 + loss_physics * 0.4

    def supervised_loss(self, pred, rho_label, loss_func):
        loss = loss_func(pred, rho_label)
        return loss
