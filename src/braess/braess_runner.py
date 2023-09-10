import numpy as np
import torch
import time

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
        (braess_loader.init_rhos[:, :, :, None]),
        braess_loader.T,
        axis=-1,
    )
    model_input = np.transpose(init_rho_copies, (0, 1, 3, 2))
    messages = np.zeros(
        (
            braess_loader.n_samples,
            braess_loader.N_edges,
            braess_loader.T,
            braess_loader.N + 1,
            braess_loader.N + 1,
            1,
        ),
        dtype=np.float32,
    )
    for sample_i in range(braess_loader.n_samples):
        messages[sample_i, :, :, :, :, 0] = np.transpose(
            all_cum_trans[sample_i], (0, 3, 1, 2)
        )

    preds = None
    for it in range(config["train"]["iterations"]):
        model_output = model(model_input, messages=messages)
        preds = torch.transpose(model_output, 3, 2)
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
        (braess_loader.terminal_Vs[:, :, :, None]),
        braess_loader.T + 1,
        axis=-1,
    )
    model_input = np.transpose(terminal_V_copies, (0, 1, 3, 2))
    messages = np.zeros(
        (
            braess_loader.n_samples,
            braess_loader.N_edges,
            braess_loader.T + 1,
            braess_loader.N + 2,
            braess_loader.N + 2,
            1,
        ),
        dtype=np.float32,
    )
    for sample_i in range(braess_loader.n_samples):
        messages[sample_i, :, :, :, :, 0] = np.transpose(
            all_cum_trans[sample_i], (0, 3, 1, 2)
        )

    preds = None
    for it in range(config["train"]["iterations"]):
        model_output = model(model_input, messages=messages)
        preds = torch.transpose(model_output, 3, 2)
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
        loss = sup_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it % 100 == 0:
            print(f"It {it}: loss {float(sup_loss)}")

    V_preds = preds[:, :, :-2, :].detach().numpy()
    return V_preds, float(sup_loss)


def run_rho_V(
    braess_loader,
    f_args,
    config,
    check_id=0,
    show=True,
):
    rho_preds = np.repeat(
        (braess_loader.init_rhos[:, :, :-1, None]), braess_loader.T, axis=-1
    )
    V_preds = np.repeat(
        (braess_loader.terminal_Vs[:, :, :-2, None]), braess_loader.T + 1, axis=-1
    )
    # rho_preds, V_preds = braess_loader.rhos, braess_loader.Vs
    u_hist, beta_hist = list(), list()
    best_rho, best_V, best_loss, best_ep = None, None, 1e8, 0
    for ep in range(config["train"]["epochs"]):
        print(f"-------------------{ep}-------------")
        start_time = time.time()
        beta_message, pi_message = braess_loader.get_beta_pi_from_V(V_preds)
        beta_hist.append(beta_message)
        beta_message = np.array(beta_hist).mean(axis=0)
        u_message = braess_loader.get_u_from_rho_V_pi(rho_preds, V_preds, pi_message)
        u_hist.append(u_message)
        u_message = np.array(u_hist).mean(axis=0)

        rho_preds, rho_loss = run_rho(
            braess_loader, u_message, rho_preds, beta_message, f_args, config
        )
        V_preds, V_loss = run_V(
            braess_loader, u_message, rho_preds, pi_message, f_args, config
        )
        if rho_loss < best_loss:
            best_rho = rho_preds
            best_V = V_preds
            best_ep = ep
            best_loss = rho_loss

        print("*** Epoch=", ep, "rho loss=", rho_loss, ", V loss=", V_loss)
        print(f"training time: {time.time() - start_time} seconds")

        if show:
            plot_4d(
                braess_loader.N,
                int(braess_loader.T / braess_loader.N),
                rho_preds[check_id],
                (0, 4, 3),
                f"pred-rho-043-{check_id}-ep{ep}",
            )
            plot_4d(
                braess_loader.N,
                int(braess_loader.T / braess_loader.N),
                V_preds[check_id, :, :, :-1],
                (0, 4, 3),
                f"pred-V-043-{check_id}-ep{ep}",
            )
            plot_4d(
                braess_loader.N,
                int(braess_loader.T / braess_loader.N),
                rho_preds[check_id],
                (1, 3),
                f"pred-rho-13-{check_id}-ep{ep}",
            )
            plot_4d(
                braess_loader.N,
                int(braess_loader.T / braess_loader.N),
                V_preds[check_id, :, :, :-1],
                (1, 3),
                f"pred-V-13-{check_id}-ep{ep}",
            )

    plot_4d(
        braess_loader.N,
        int(braess_loader.T / braess_loader.N),
        best_rho[check_id],
        (0, 4, 3),
        f"pred-rho-043-{check_id}-ep{best_ep}",
    )
    plot_4d(
        braess_loader.N,
        int(braess_loader.T / braess_loader.N),
        best_V[check_id, :, :, :-1],
        (0, 4, 3),
        f"pred-V-043-{check_id}-ep{best_ep}",
    )
    plot_4d(
        braess_loader.N,
        int(braess_loader.T / braess_loader.N),
        best_rho[check_id],
        (1, 3),
        f"pred-rho-13-{check_id}-ep{best_ep}",
    )
    plot_4d(
        braess_loader.N,
        int(braess_loader.T / braess_loader.N),
        best_V[check_id, :, :, :-1],
        (1, 3),
        f"pred-V-13-{check_id}-ep{best_ep}",
    )
