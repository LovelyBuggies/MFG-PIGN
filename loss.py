import torch


def transition_loss(pred, transitions, init_rho, loss_func):
    N, T = pred.shape
    loss_ic = loss_func(
        torch.matmul(torch.from_numpy(transitions[0]), pred[:, 0]),
        torch.from_numpy(init_rho),
    )
    loss_physics = 0.0
    for t in range(1, T):
        pred_t = torch.matmul(torch.from_numpy(transitions[t]), pred[:, t - 1])
        loss_physics += loss_func(pred_t, pred[:, t])

    return loss_ic * 0.6 + loss_physics * 0.4


def supervised_loss(pred, rho_label, loss_func):
    loss = loss_func(pred, rho_label)
    return loss
