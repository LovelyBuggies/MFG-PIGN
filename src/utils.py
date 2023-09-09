import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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


def plot_3d(n_x, n_t, matrix, ax_name, fig_name=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection="3d")
    x = np.linspace(0, 1, n_x)
    t = np.linspace(0, 1, n_t)
    t_mesh, x_mesh = np.meshgrid(t, x)
    surf = ax.plot_surface(
        x_mesh, t_mesh, matrix, cmap=cm.jet, linewidth=0, antialiased=False
    )
    ax.grid(False)
    ax.tick_params(axis="both", which="major", labelsize=18, pad=10)

    ax.set_xlabel(r"$x$", fontsize=24, labelpad=20)
    ax.set_xlim(min(x), max(x))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    plt.ylabel(r"$t$", fontsize=24, labelpad=20)
    ax.set_ylim(min(t), max(t))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    ax.set_zlabel(ax_name, fontsize=24, labelpad=20, rotation=90)
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.view_init(elev=25, azim=-128)
    if not fig_name:
        plt.show()
    else:
        plt.savefig(fig_name, bbox_inches="tight")


def plot_4d(n_cell, T_terminal, matrix, concat, ax_name, fig_name=None):
    matrix_new = matrix[concat[0], :, :]
    for i in range(1, len(concat)):
        matrix_new = np.append(matrix_new, matrix[concat[i], :, :], axis=0)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    x = np.linspace(0, len(concat), len(concat) * n_cell)
    t = np.linspace(0, T_terminal, n_cell * T_terminal)
    t_mesh, x_mesh = np.meshgrid(t, x)
    surf = ax.plot_surface(
        x_mesh, t_mesh, matrix_new, cmap=cm.jet, linewidth=0, antialiased=False
    )
    ax.set_zlabel(ax_name, fontsize=24, labelpad=20, rotation=90)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    plt.xlim(max(x), min(x))
    if not fig_name:
        plt.show()
    else:
        plt.savefig(fig_name)


def get_args_kwargs(f_config):
    device = torch.device("cpu")
    f_args = (
        f_config["input_dim"],
        f_config["output_dim"],
        f_config["hidden_num"] - 2,
        f_config["fc_dim"],
    )
    f_kwargs = {
        "activation_type": f_config["activation_type"],
        "last_activation_type": f_config["last_activation_type"],
        "device": device,
    }
    if f_config["hidden_num"] < 2:
        raise ValueError("Cannot have 'hidden_num' < 2")

    return f_args, f_kwargs
