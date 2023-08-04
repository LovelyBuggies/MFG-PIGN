import torch


def transition_loss(preds, all_trans, rho_inits, loss_kwargs):
    n_samples, N, T = preds.shape
    loss_ic = 0.0
    loss_physics = 0.0
    loss_func = loss_kwargs["func"]
    for sample_i in range(n_samples):
        loss_ic += loss_func(
            torch.matmul(
                torch.from_numpy(all_trans[sample_i][0]), preds[sample_i, :, 0]
            ),
            torch.from_numpy(rho_inits[sample_i]),
        )
        for t in range(1, T):
            pred_t = torch.matmul(
                torch.from_numpy(all_trans[sample_i][t]),
                preds[sample_i, :, t - 1],
            )
            loss_physics += loss_func(pred_t, preds[sample_i, :, t])

    return loss_ic * loss_kwargs["w_ic"] + loss_physics * loss_kwargs["w_physics"]


def supervised_loss(preds, rho_labels, loss_kwargs):
    loss_func = loss_kwargs["func"]
    loss = loss_func(preds, rho_labels)
    return loss
