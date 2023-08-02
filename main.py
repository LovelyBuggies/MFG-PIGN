import numpy as np
import torch
from loader import RingRoadLoader
from model import PIGN
from loss import supervised_loss, transition_loss
from utils import plot_3d


if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        DEVICE = torch.device("cpu")

    """Loader"""
    rho_label = np.loadtxt("data/rho.csv", delimiter=",", dtype=np.float32)
    u_label = np.loadtxt("data/u.csv", delimiter=",", dtype=np.float32)
    V_label = np.loadtxt("data/V.csv", delimiter=",", dtype=np.float32)
    ring_data_loader = RingRoadLoader(rho_label, u_label, V_label)

    """Hyper-params"""
    f_x_args = (2, 1, 3, 32)
    f_x_kwargs = {
        "activation_type": "none",
        "last_activation_type": "none",
        "device": DEVICE,
    }
    f_m_args = (1, 1, 3, 32)
    f_m_kwargs = {
        "activation_type": "none",
        "last_activation_type": "none",
        "device": DEVICE,
    }
    f_j_args = (ring_data_loader.N, 1, 3, 32)
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
        ring_data_loader.A,
    )
    optimizer_kwargs = {"lr": 0.001}
    optimizer = torch.optim.Adam(
        [p for p in pign.parameters() if p.requires_grad is True], **optimizer_kwargs
    )
    loss_func = torch.nn.MSELoss()

    """Params"""
    transitions, cumulative_transitions = ring_data_loader.get_transition_matrix()
    x_pign = np.repeat((ring_data_loader.init_rho[:, None]), ring_data_loader.T, axis=1)
    x_pign = np.transpose(x_pign, (1, 0))
    message = np.zeros(
        (ring_data_loader.T, ring_data_loader.N, ring_data_loader.N, 1),
        dtype=np.float32,
    )
    for t in range(ring_data_loader.T):
        message[t, :, :, 0] = cumulative_transitions[t]

    """Train"""
    for it in range(500):
        # forward input as (T, X)
        pred = pign(x_pign, message=message)
        loss = transition_loss(
            torch.transpose(pred, 1, 0),
            transitions,
            ring_data_loader.init_rho,
            loss_func,
        )
        # loss = supervised_loss(
        #     torch.transpose(pred, 1, 0),
        #     torch.from_numpy(rho_label),
        #     loss_func,
        # )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("it=", it, "loss=", float(loss))

    final_pred = torch.transpose(pred, 1, 0)
    plot_3d(8, 8, final_pred.detach().numpy(), r"$\rho$")
