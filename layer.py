import torch
from torch import nn
import torch.nn.functional as F


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


@torch.jit.script
def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixy,ioxy->boxy", a, b)
    return res


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[2, 3])

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
        return x


def add_padding2(x, num_pad1, num_pad2):
    if max(num_pad1) > 0 or max(num_pad2) > 0:
        res = F.pad(
            x, (num_pad2[0], num_pad2[1], num_pad1[0], num_pad1[1]), "constant", 0.0
        )
    else:
        res = x
    return res


def remove_padding2(x, num_pad1, num_pad2):
    if max(num_pad1) > 0 or max(num_pad2) > 0:
        res = x[..., num_pad1[0] : -num_pad1[1], num_pad2[0] : -num_pad2[1]]
    else:
        res = x
    return res


def _get_act(act):
    if act == "tanh":
        func = F.tanh
    elif act == "gelu":
        func = F.gelu
    elif act == "relu":
        func = F.relu_
    elif act == "elu":
        func = F.elu_
    elif act == "leaky_relu":
        func = F.leaky_relu_
    else:
        raise ValueError(f"{act} is not supported")
    return func
