import numpy as np
import torch

from .model import PIGN_rho, PIGN_V
from .loss import supervised_loss, transition_loss_rho, transition_loss_V
from .test import all_trans_tester_rho, all_trans_tester_V
from .utils import plot_3d


def run_rho(
    ring_loader, u_message, args, config, epoch, check_id, test=True, show=True
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

    """Params"""
    all_trans, all_cum_trans = ring_loader.get_trans_matrix_rho(u_message)
    if test:
        all_trans_tester_rho(ring_loader, all_trans, all_cum_trans, check_id)

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

    """Train"""
    preds = None
    for it in range(epoch):
        # forward input as (T, X)
        model_output = model(model_input, messages=messages)
        preds = torch.transpose(model_output, 2, 1)
        loss = 0 * supervised_loss(
            preds,
            torch.from_numpy(ring_loader.rhos),
            loss_kwargs,
        ) + 1 * transition_loss_rho(
            preds,
            all_trans,
            ring_loader.init_rhos,
            loss_kwargs,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("it=", it, "loss=", float(loss))

    if show:
        plot_3d(
            ring_loader.N,
            ring_loader.T,
            ring_loader.rhos[check_id],
            f"truth-{check_id}",
        )
        plot_3d(
            ring_loader.N,
            ring_loader.T,
            preds[check_id].detach().numpy(),
            f"pred-{check_id}",
        )

    return preds


def run_V(
    ring_loader,
    u_message,
    rho_message,
    args,
    config,
    epoch,
    check_id,
    test=True,
    show=True,
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

    """Params"""
    all_trans, all_cum_trans = ring_loader.get_trans_matrix_V(u_message, rho_message)
    if test:
        all_trans_tester_V(ring_loader, all_trans, all_cum_trans, check_id)

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

    """Train"""
    preds = None
    for it in range(epoch):
        # forward input as (T, X)
        model_output = model(model_input, messages=messages)
        preds = torch.transpose(model_output, 2, 1)
        loss = 0 * supervised_loss(
            preds[:, :-1, :],
            torch.from_numpy(ring_loader.Vs[:, :-1, :]),
            loss_kwargs,
        ) + 1 * transition_loss_V(
            preds,
            all_trans,
            ring_loader.terminal_Vs,
            loss_kwargs,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("it=", it, "loss=", float(loss))

    V_preds = preds.detach().numpy()
    V_preds[:, -1, :] = V_preds[:, 0, :]
    if show:
        plot_3d(
            ring_loader.N + 1,
            ring_loader.T + 1,
            ring_loader.Vs[check_id],
            f"truth-{check_id}",
        )
        plot_3d(
            ring_loader.N + 1, ring_loader.T + 1, V_preds[check_id], f"pred-{check_id}"
        )

    return V_preds


def run_rho_V(ring_loader, args, config, epoch, check_id, show=True):
    rho_preds = 0.5 * np.ones(
        (ring_loader.n_samples, ring_loader.N, ring_loader.T), dtype=np.float32
    )
    V_preds = np.zeros(
        (ring_loader.n_samples, ring_loader.N + 1, ring_loader.T + 1), dtype=np.float32
    )
    rho_labels = ring_loader.rhos
    V_labels = ring_loader.Vs
    u_labels = ring_loader.us
    inner_epoch = 500
    epoch = 1
    for it in range(epoch):
        print(f"-------- Epoch: {it} --------\n")
        # u_message = ring_loader.get_u_from_rho_V(rho_preds, V_preds)
        # rho_message = rho_preds  # or get_rho_from_u
        u_message = u_labels
        rho_message = rho_labels
        rho_preds = run_rho(
            ring_loader,
            u_message,
            args,
            config,
            epoch=inner_epoch,
            check_id=check_id,
            test=False,
            show=True,
        )
        V_preds = run_V(
            ring_loader,
            u_message,
            rho_message,
            args,
            config,
            epoch=inner_epoch,
            check_id=check_id,
            test=False,
            show=True,
        )

    if show:
        plot_3d(
            ring_loader.N,
            ring_loader.T,
            rho_preds[check_id].detach().numpy(),
            f"pred-{check_id}",
        )
        plot_3d(
            ring_loader.N + 1, ring_loader.T + 1, V_preds[check_id], f"pred-{check_id}"
        )
