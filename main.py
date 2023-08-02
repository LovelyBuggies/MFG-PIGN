import numpy as np
import scipy.io
import torch
from loader import RingRoadLoader
from model import PIGN
from loss import supervised_loss, transition_loss
from utils import plot_3d
from test import all_transition_tester


if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        DEVICE = torch.device("cpu")

    """Loader"""
    SAMPLE_NUM = 1
    WANNA_CHECK = 0
    rho_labels = scipy.io.loadmat("data/data-classic.mat")["rhos"]
    u_labels = scipy.io.loadmat("data/data-classic.mat")["us"]
    V_labels = scipy.io.loadmat("data/data-classic.mat")["Vs"]
    rho_labels = np.array(rho_labels, dtype=np.float32)[:SAMPLE_NUM, :, :]
    u_labels = np.array(u_labels, dtype=np.float32)[:SAMPLE_NUM, :, :]
    V_labels = np.array(V_labels, dtype=np.float32)[:SAMPLE_NUM, :, :]
    ring_loader = RingRoadLoader(rho_labels, u_labels, V_labels)

    """Hyper-params"""
    f_x_args = (2, 1, 5, 32)
    f_x_kwargs = {
        "activation_type": "none",
        "last_activation_type": "none",
        "device": DEVICE,
    }
    f_m_args = (1, 1, 5, 32)
    f_m_kwargs = {
        "activation_type": "none",
        "last_activation_type": "none",
        "device": DEVICE,
    }
    f_j_args = (ring_loader.N, 1, 5, 32)
    f_j_kwargs = {
        "activation_type": "none",
        "last_activation_type": "none",
        "device": DEVICE,
    }
    pign = PIGN(
        f_m_args,
        f_m_kwargs,
        f_j_args,
        f_j_kwargs,
        f_x_args,
        f_x_kwargs,
        ring_loader.A,
    )
    optimizer_kwargs = {"lr": 0.001}
    optimizer = torch.optim.Adam(
        [p for p in pign.parameters() if p.requires_grad is True], **optimizer_kwargs
    )
    loss_func = torch.nn.MSELoss()

    """Params"""
    all_transitions, all_cumulative_transitions = ring_loader.get_transition_matrix()
    all_transition_tester(
        ring_loader, all_transitions, all_cumulative_transitions, WANNA_CHECK
    )

    x_pign = np.repeat((ring_loader.init_rhos[:, :, None]), ring_loader.T, axis=-1)
    x_pign = np.transpose(x_pign, (0, 2, 1))
    messages = np.zeros(
        (ring_loader.n_samples, ring_loader.T, ring_loader.N, ring_loader.N, 1),
        dtype=np.float32,
    )
    for sample_i in range(ring_loader.n_samples):
        for t in range(ring_loader.T):
            messages[sample_i, t, :, :, 0] = all_cumulative_transitions[sample_i][t]

    """Train"""
    for it in range(500):
        # forward input as (T, X)
        preds = pign(x_pign, messages=messages)
        # loss = supervised_loss(
        #     torch.transpose(preds, 2, 1),
        #     torch.from_numpy(rho_labels),
        #     loss_func,
        # )
        loss = transition_loss(
            torch.transpose(preds, 2, 1),
            all_transitions,
            ring_loader.init_rhos,
            loss_func,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("it=", it, "loss=", float(loss))

    final_preds = torch.transpose(preds, 2, 1)
    plot_3d(8, 8, final_preds[WANNA_CHECK].detach().numpy(), r"$\rho$")
