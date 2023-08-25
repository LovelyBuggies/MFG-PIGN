import numpy as np
import torch

from src.braess.braess_model import PIGN_rho


def run_rho(
    braess_loader,
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

    all_trans, all_cum_trans = braess_loader.get_trans_matrix_rho(u_message)

    init_rho_copies = np.repeat(
        (braess_loader.init_rhos[:, :, None]), braess_loader.T, axis=-1
    )
    model_input = np.transpose(init_rho_copies, (0, 2, 1))
    messages = np.zeros(
        (braess_loader.n_samples, braess_loader.T, braess_loader.N, braess_loader.N, 1),
        dtype=np.float32,
    )
    for sample_i in range(braess_loader.n_samples):
        for t in range(braess_loader.T):
            messages[sample_i, t, :, :, 0] = all_cum_trans[sample_i][t]

    preds = None
    for it in range(config["train"]["iterations"]):
        model_output = model(model_input, messages=messages)
        preds = torch.transpose(model_output, 2, 1)
        sup_loss = supervised_loss(
            preds,
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

    rho_preds = preds.detach().numpy()
    return rho_preds, float(sup_loss)
