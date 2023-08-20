import numpy as np
import torch

from src.model import PIGN_rho, PIGN_V
from src.loss import supervised_loss, transition_loss_rho, transition_loss_V
from src.test import all_trans_tester_rho, all_trans_tester_V
from src.utils import plot_3d, get_u_from_rho_V


def run_rho(
    ring_loader,
    u_message,
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

    all_trans, all_cum_trans = ring_loader.get_trans_matrix_rho(u_message)

    init_rho_copies = np.repeat(
        (ring_loader.init_rhos[:, :, None]), ring_loader.T, axis=-1
    )
    model_input = np.transpose(init_rho_copies, (0, 2, 1))
    messages = np.zeros(
        (ring_loader.n_samples, ring_loader.T, ring_loader.N, ring_loader.N, 1),
        dtype=np.float32,
    )
    for sample_i in range(ring_loader.n_samples):
        for t in range(ring_loader.T):
            messages[sample_i, t, :, :, 0] = all_cum_trans[sample_i][t]

    preds = None
    for it in range(config["train"]["epochs"]):
        model_output = model(model_input, messages=messages)
        preds = torch.transpose(model_output, 2, 1)
        sup_loss = supervised_loss(
            preds,
            torch.from_numpy(ring_loader.rhos),
            loss_kwargs,
        )
        tran_loss = transition_loss_rho(
            preds,
            all_trans,
            ring_loader.init_rhos,
            loss_kwargs,
        )
        loss = tran_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    rho_preds = preds.detach().numpy()
    return rho_preds, float(sup_loss)


def run_V(
    ring_loader,
    u_message,
    rho_message,
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

    all_trans, all_cum_trans = ring_loader.get_trans_matrix_V(u_message, rho_message)
    terminal_V_copies = np.repeat(
        (ring_loader.terminal_Vs[:, :, None]), ring_loader.T + 1, axis=-1
    )
    model_input = np.transpose(terminal_V_copies, (0, 2, 1))
    messages = np.zeros(
        (
            ring_loader.n_samples,
            ring_loader.T + 1,
            ring_loader.N + 1,
            ring_loader.N + 1,
            1,
        ),
        dtype=np.float32,
    )
    for sample_i in range(ring_loader.n_samples):
        for t in range(ring_loader.T + 1):
            messages[sample_i, t, :, :, 0] = all_cum_trans[sample_i][t]

    preds = None
    for it in range(config["train"]["epochs"]):
        # forward input as (T, X)
        model_output = model(model_input, messages=messages)
        preds = torch.transpose(model_output, 2, 1)
        sup_loss = supervised_loss(
            preds[:, :-1, :],
            torch.from_numpy(ring_loader.Vs[:, :-1, :]),
            loss_kwargs,
        )
        tran_loss = transition_loss_V(
            preds,
            all_trans,
            ring_loader.terminal_Vs,
            loss_kwargs,
        )
        loss = tran_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    V_preds = preds.detach().numpy()
    V_preds[:, -1, :] = V_preds[:, 0, :]

    return V_preds, float(sup_loss)


def run_rho_V(ring_loader, args, config, check_id, show=True):
    rho_labels = ring_loader.rhos
    V_labels = ring_loader.Vs
    rho_preds = np.repeat(ring_loader.init_rhos[:, :, None], ring_loader.T, axis=-1)
    V_preds = np.repeat(ring_loader.terminal_Vs[:, :, None], ring_loader.T + 1, axis=-1)
    V_preds[:, -1, :] = V_preds[:, 0, :]
    # rho_preds = rho_labels
    # V_preds = V_labels
    epoch = 50
    u_hist, rho_hist = list(), list()
    best_rho, best_V, best_loss, best_ep = None, None, 1e8, 0
    for ep in range(epoch):
        u_message = get_u_from_rho_V(rho_preds, V_preds)
        u_hist.append(u_message)
        u_message = np.array(u_hist).mean(axis=0)
        # u_message = ring_loader.us
        rho_message = rho_preds
        # rho_hist.append(rho_message)
        # rho_message = np.array(rho_hist).mean(axis=0)

        rho_preds, rho_loss = run_rho(
            ring_loader,
            u_message,
            args,
            config,
        )
        V_preds, V_loss = run_V(
            ring_loader,
            u_message,
            rho_message,
            args,
            config,
        )
        if rho_loss < best_loss:
            best_rho = rho_preds
            best_V = V_preds
            best_ep = ep
            best_loss = rho_loss
        print("*** Epoch=", ep, "rho loss=", rho_loss, ", V loss=", V_loss)

    if show:
        plot_3d(
            ring_loader.N,
            ring_loader.T,
            best_rho[check_id],
            f"pred-rho-{check_id}-ep{best_ep}",
        )
        plot_3d(
            ring_loader.N,
            ring_loader.T,
            ring_loader.rhos[check_id],
            f"label-rho-{check_id}",
        )
        plot_3d(
            ring_loader.N + 1,
            ring_loader.T + 1,
            best_V[check_id],
            f"pred-V-{check_id}-ep{best_ep}",
        )
        plot_3d(
            ring_loader.N + 1,
            ring_loader.T + 1,
            ring_loader.Vs[check_id],
            f"label-V-{check_id}",
        )
