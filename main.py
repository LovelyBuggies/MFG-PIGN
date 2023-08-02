import numpy as np
import torch
from loader import RingRoadLoader
from model import MPNN
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
    rho_inits = np.repeat(
        (ring_data_loader.init_rho[:, None]), ring_data_loader.T, axis=1
    )

    """Get transitions"""
    transitions, cumulative_transitions = ring_data_loader.get_transition_matrix()

    """Test unit"""
    # rho = np.zeros(rho_label.shape, dtype=np.float32)
    # for t in range(ring_data_loader.T):
    #     rho[:, t] = np.dot(cumulative_transitions[t], rho_inits[:, t])
    # plot_3d(8, 8, rho, "pre")

    """Train model"""
    f_x_args = (2, 1, 3, 32)
    f_x_kwargs = {
        "activation_type": "none",
        "last_activation_type": "none",
        "device": DEVICE,
    }
    f_m_args = (3, 1, 3, 32)
    f_m_kwargs = {
        "activation_type": "none",
        "last_activation_type": "none",
        "device": DEVICE,
    }
    f_new_args = (ring_data_loader.N, 1, 3, 32)
    f_new_kwargs = {
        "activation_type": "none",
        "last_activation_type": "none",
        "device": DEVICE,
    }
    mpnn = MPNN(
        f_x_args,
        f_x_kwargs,
        f_m_args,
        f_m_kwargs,
        f_new_args,
        f_new_kwargs,
        ring_data_loader.A,
    )
    optimizer_kwargs = {"lr": 0.001}
    optimizer = torch.optim.Adam(
        [p for p in mpnn.parameters() if p.requires_grad is True], **optimizer_kwargs
    )
    loss_func = torch.nn.MSELoss()
    for it in range(500):
        pred = mpnn(
            np.transpose(rho_inits, (1, 0)), cumulative_transitions
        )  # forward is (T, X) pred
        # loss = mpnn.transition_loss(torch.transpose(pred, 1, 0), transitions, loss_func)
        loss = mpnn.supervised_loss(
            torch.transpose(pred, 1, 0), torch.from_numpy(rho_label), loss_func
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("it=", it, "loss=", float(loss))

    pred = (
        torch.transpose(
            mpnn(np.transpose(rho_inits, (1, 0)), cumulative_transitions), 1, 0
        )
        .detach()
        .numpy()
    )
    plot_3d(8, 8, pred, "pre")
