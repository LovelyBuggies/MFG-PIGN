import numpy as np
import torch

from src.braess.braess_model import PIGN_rho, PIGN_V
from src.braess.braess_loss import (
    supervised_loss,
    transition_loss_rho,
    transition_loss_V,
)
from src.utils import plot_4d


def run_rho(
    braess_loader,
    u_message,
    rho_message,
    beta_message,
    args,
    config,
):
    model = PIGN_rho(*args)
    optimizer_kwargs = {"lr": config["train"]["lr"]}
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad is True], **optimizer_kwargs
    )
    loss_kwargs = {
        "func": torch.nn.MSELoss(),
        "w_ic": config["train"]["w_ic"],
        "w_physics": config["train"]["w_physics"],
    }

    all_trans, all_cum_trans = braess_loader.get_trans_matrix_rho(
        u_message, rho_message, beta_message
    )
    init_rho_copies = np.repeat(
        (braess_loader.init_rhos.reshape(braess_loader.n_samples, -1)[:, :, None]),
        braess_loader.T,
        axis=-1,
    )
    model_input = np.transpose(init_rho_copies, (0, 2, 1))
    l_trans = braess_loader.N + 1
    l_large_trans = l_trans * braess_loader.N_edges
    messages = np.zeros(
        (
            braess_loader.n_samples,
            braess_loader.T,
            l_large_trans,
            l_large_trans,
            1,
        ),
        dtype=np.float32,
    )
    for sample_i in range(braess_loader.n_samples):
        for t in range(braess_loader.T):
            for i in range(l_large_trans):
                edge_i = int(i / l_trans)
                for j in range(edge_i * l_trans, (edge_i + 1) * l_trans):
                    messages[sample_i, t, i, j, 0] = all_cum_trans[sample_i][
                        edge_i, i % l_trans, j % l_trans, t
                    ]

    preds = None
    for it in range(config["train"]["iterations"]):
        model_output = model(model_input, messages=messages)
        preds = torch.transpose(model_output, 2, 1).reshape(
            braess_loader.n_samples, braess_loader.N_edges, l_trans, braess_loader.T
        )
        sup_loss = supervised_loss(
            preds[:, :, :-1, :],
            torch.from_numpy(braess_loader.rhos),
            loss_kwargs,
        )
        tran_loss = transition_loss_rho(
            preds,
            all_trans,
            braess_loader.init_rhos,
            loss_kwargs,
        )
        loss = tran_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it % 100 == 0:
            print(f"It {it}: loss {float(sup_loss)}")
        if float(sup_loss) < 1e-6:
            break

    rho_preds = preds[:, :, :-1, :].detach().numpy()
    return rho_preds, float(sup_loss)


def run_V(
    braess_loader,
    u_message,
    rho_message,
    pi_message,
    args,
    config,
):
    model = PIGN_V(*args)
    optimizer_kwargs = {"lr": config["train"]["lr"]}
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad is True], **optimizer_kwargs
    )
    loss_kwargs = {
        "func": torch.nn.MSELoss(),
        "w_ic": config["train"]["w_ic"],
        "w_physics": config["train"]["w_physics"],
    }

    all_trans, all_cum_trans = braess_loader.get_trans_matrix_V(
        u_message, rho_message, pi_message
    )
    terminal_V_copies = np.repeat(
        (braess_loader.terminal_Vs.reshape(braess_loader.n_samples, -1)[:, :, None]),
        braess_loader.T + 1,
        axis=-1,
    )
    model_input = np.transpose(terminal_V_copies, (0, 2, 1))
    l_trans = braess_loader.N + 2
    l_large_trans = l_trans * braess_loader.N_edges
    messages = np.zeros(
        (
            braess_loader.n_samples,
            braess_loader.T + 1,
            l_large_trans,
            l_large_trans,
            1,
        ),
        dtype=np.float32,
    )
    for sample_i in range(braess_loader.n_samples):
        for t in range(braess_loader.T + 1):
            for i in range(l_large_trans):
                edge_i = int(i / l_trans)
                for j in range(edge_i * l_trans, (edge_i + 1) * l_trans):
                    messages[sample_i, t, i, j, 0] = all_cum_trans[sample_i][
                        edge_i, i % l_trans, j % l_trans, t
                    ]

    preds = None
    for it in range(config["train"]["iterations"]):
        model_output = model(model_input, messages=messages)
        preds = torch.transpose(model_output, 2, 1).reshape(
            braess_loader.n_samples, braess_loader.N_edges, l_trans, braess_loader.T + 1
        )
        sup_loss = supervised_loss(
            preds[:, :, :-2, :],
            torch.from_numpy(braess_loader.Vs),
            loss_kwargs,
        )
        tran_loss = transition_loss_V(
            preds,
            all_trans,
            braess_loader.terminal_Vs,
            loss_kwargs,
        )
        loss = tran_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it % 100 == 0:
            print(f"It {it}: loss {float(sup_loss)}")
        if float(sup_loss) < 1e-6:
            break

    V_preds = preds[:, :, :-2, :].detach().numpy()
    return V_preds, float(sup_loss)
