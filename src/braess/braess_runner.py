import numpy as np
import torch

from src.braess.braess_model import PIGN_rho
from src.braess.braess_loss import supervised_loss, transition_loss_rho
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
        (braess_loader.init_rhos[:, :, :, None]), braess_loader.T, axis=-1
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
        # tran_loss = transition_loss_rho(
        #     preds,
        #     all_trans,
        #     braess_loader.init_rhos,
        #     loss_kwargs,
        # )
        loss = sup_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"It {it}: loss {float(sup_loss)}")

    rho_preds = preds[:, :, :-1, :].detach().numpy()
    return rho_preds, float(sup_loss)
