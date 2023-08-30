import torch


def supervised_loss(preds, labels, loss_kwargs):
    loss_func = loss_kwargs["func"]
    loss = loss_func(preds, labels)
    return loss


def transition_loss_rho(preds, all_trans, rho_inits, loss_kwargs):
    n_samples, N_edges, N, T = preds.shape
    loss_ic = 0.0
    loss_physics = 0.0
    loss_func = loss_kwargs["func"]
    for sample_i in range(n_samples):
        for edge_i in range(N_edges):
            loss_ic += loss_func(
                torch.matmul(
                    torch.from_numpy(all_trans[sample_i][edge_i, :, :, 0]),
                    preds[sample_i, edge_i, :, 0],
                ),
                torch.from_numpy(rho_inits[sample_i, edge_i]),
            )
            for t in range(1, T):
                pred_t = torch.matmul(
                    torch.from_numpy(all_trans[sample_i][edge_i, :, :, t]),
                    preds[sample_i, edge_i, :, t - 1],
                )
                loss_physics += loss_func(pred_t, preds[sample_i, edge_i, :, t])

    return loss_ic * loss_kwargs["w_ic"] + loss_physics * loss_kwargs["w_physics"]


def transition_loss_V(preds, all_trans, V_terminals, loss_kwargs):
    n_samples, N_edges, N, T = preds.shape
    N, T = N - 1, T - 1
    loss_ic = 0.0
    loss_physics = 0.0
    loss_func = loss_kwargs["func"]
    for sample_i in range(n_samples):
        for edge_i in range(N_edges):
            loss_ic += loss_func(
                torch.matmul(
                    torch.from_numpy(all_trans[sample_i][edge_i, :, :, -1]),
                    preds[sample_i, edge_i, :, -1],
                ),
                torch.from_numpy(V_terminals[sample_i, edge_i]),
            )
            for t in range(T - 1, -1, -1):
                pred_t = torch.matmul(
                    torch.from_numpy(all_trans[sample_i][edge_i, :, :, t]),
                    preds[sample_i, edge_i, :, t - 1],
                )
                loss_physics += loss_func(pred_t, preds[sample_i, edge_i, :, t])

    return loss_ic * loss_kwargs["w_ic"] + loss_physics * loss_kwargs["w_physics"]
