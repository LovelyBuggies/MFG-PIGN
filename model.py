import numpy as np
import torch
from torch import nn
from layer import (
    get_fully_connected_layer,
    SpectralConv2d,
    add_padding2,
    remove_padding2,
    _get_act,
)


class MLP(nn.Module):
    def __init__(self, nn_args, nn_kwargs):
        super().__init__()
        self.model = get_fully_connected_layer(*nn_args, **nn_kwargs)
        self.device = nn_kwargs["device"]

    def forward(self, x):
        return self.model(x)


class FNO(nn.Module):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        fc_dim=128,
        layers=None,
        in_dim=3,
        out_dim=1,
        act="gelu",
        pad_ratio=[0.0, 0.0],
    ):
        super(FNO, self).__init__()
        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, "Cannot add padding in more than 2 directions"
        self.modes1 = modes1
        self.modes2 = modes2

        self.pad_ratio = pad_ratio
        if layers is None:
            self.layers = [width] * (len(modes1) + 1)
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv2d(in_size, out_size, mode1_num, mode2_num)
                for in_size, out_size, mode1_num, mode2_num in zip(
                    self.layers, self.layers[1:], self.modes1, self.modes2
                )
            ]
        )

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, layers[-1])
        self.fc3 = nn.Linear(layers[-1], out_dim)
        self.act = _get_act(act)

    def forward(self, x):
        size_1, size_2 = x.shape[1], x.shape[2]
        if max(self.pad_ratio) > 0:
            num_pad1 = [round(i * size_1) for i in self.pad_ratio]
            num_pad2 = [round(i * size_2) for i in self.pad_ratio]
        else:
            num_pad1 = num_pad2 = [0.0]

        length = len(self.ws)
        batchsize = x.shape[0]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = add_padding2(x, num_pad1, num_pad2)
        size_x, size_y = x.shape[-2], x.shape[-1]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(
                batchsize, self.layers[i + 1], size_x, size_y
            )
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)

        x = remove_padding2(x, num_pad1, num_pad2)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        # x = self.act(x)
        x = self.fc2(x)
        # x = self.act(x)
        x = self.fc3(x)
        return x


class PIGNO(nn.Module):
    def __init__(
        self,
        ring_road,
        f_x_args,
        f_x_kwargs,
        f_m_args,
        f_m_kwargs,
    ):
        super().__init__()
        self.n_sample = ring_road.n_sample
        self.N = ring_road.N
        self.T = ring_road.T
        self.A = torch.from_numpy(ring_road.A)
        self.f_x = MLP(f_x_args, f_x_kwargs)
        self.f_m = MLP(f_m_args, f_m_kwargs)
        self.f_pino = FNO(
            modes1=[5, 5, 5, 5],
            modes2=[5, 5, 5, 5],
            fc_dim=64,
            layers=[16, 24, 24, 32, 32],
            act="gelu",
        )

    def forward(self, rho_init, V_terminal):
        rho_init, V_terminal = torch.Tensor(rho_init), torch.Tensor(V_terminal)
        Xs = torch.cat((rho_init, V_terminal), dim=1)
        Xs_len = Xs.shape[1]
        gridx = torch.tensor(np.linspace(0, 1, Xs_len + 1)[:-1], dtype=torch.float)
        gridt = torch.tensor(np.linspace(0, 1, self.T), dtype=torch.float)
        gridx = gridx.reshape(1, 1, Xs_len)
        gridt = gridt.reshape(1, self.T, 1)
        Xs = Xs.reshape(self.n_sample, 1, Xs_len).repeat([1, self.T, 1])
        Xs = torch.stack(
            [
                Xs,
                gridx.repeat([self.n_sample, self.T, 1]),
                gridt.repeat([self.n_sample, 1, Xs_len]),
            ],
            dim=3,
        )
        # out is the (x, t) transpose of cat(rho, V)
        out = self.f_pino(Xs)[:, :, :, 0]  # 1 * 8 * 17
        return out

    def mpnn_helper(self, out):
        mpnn_out = out.clone()
        for batch_i in range(out.shape[0]):
            rho_out_T = out[batch_i, :, : self.N]
            V_out_T = out[batch_i, :, self.N : self.N * 2]
            rho_i = rho_out_T.unsqueeze(1).repeat(1, self.N, 1)
            rho_j = rho_out_T.unsqueeze(2).repeat(1, 1, self.N)
            V_i = V_out_T.unsqueeze(1).repeat(1, self.N, 1)
            V_j = V_out_T.unsqueeze(2).repeat(1, 1, self.N)
            e_ij = torch.cat(
                [
                    (rho_i + rho_j).unsqueeze(-1) / 2,
                    (V_i + V_j).unsqueeze(-1) / 2,
                ],
                dim=-1,
            )
            message_embed = self.f_m(e_ij.to(self.f_m.device))
            message_sum = torch.sum(
                message_embed[:, :, :, 0] * self.A[batch_i, :, :], 2
            )
            torch_x = torch.concat(
                (rho_out_T[:, :, None], message_sum[:, :, None]), axis=-1
            )
            f_x_out = self.f_x(torch_x.float())
            mpnn_out[batch_i, :, :-1] = f_x_out.reshape(self.T, -1)

        return mpnn_out
