import numpy as np
import torch
from loader import RingRoad
from model import PIGNO
from loss import PINO_loss
from utils import plot_3d


if __name__ == "__main__":
    INPUT_DIM = 2
    OUTPUT_DIM = 2
    N_HIDDEN = 0
    HIDDEN_DIM = 1
    early_stop = 500

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        DEVICE = torch.device("cpu")

    rho = np.loadtxt("data/rho.csv", delimiter=",")[None, :, :]
    V = np.loadtxt("data/V.csv", delimiter=",")[None, :, :]
    ring_road = RingRoad(rho, V)

    f_x_args = (INPUT_DIM, OUTPUT_DIM, N_HIDDEN, HIDDEN_DIM)
    f_x_kwargs = {
        "activation_type": "none",
        "last_activation_type": "none",
        "device": DEVICE,
        "mean": np.array([rho.mean(), 0], dtype=np.float32),
        "std": np.array([rho.std(), 1], dtype=np.float32),
    }
    f_m_args = (INPUT_DIM, 1, N_HIDDEN, HIDDEN_DIM)
    f_m_kwargs = {
        "activation_type": "none",
        "last_activation_type": "none",
        "device": DEVICE,
        "mean": np.array([rho.mean(), V.mean()], dtype=np.float32),
        "std": np.array([rho.std(), V.std()], dtype=np.float32),
    }
    pigno = PIGNO(
        ring_road,
        f_x_args,
        f_x_kwargs,
        f_m_args,
        f_m_kwargs,
    )
    optimizer_kwargs = {"lr": 0.001}
    optimizer = torch.optim.Adam(
        [p for p in pigno.parameters() if p.requires_grad is True], **optimizer_kwargs
    )
    loss_fun = torch.nn.MSELoss()

    y = torch.Tensor(np.concatenate((ring_road.rho, ring_road.V[:, :, :-1]), axis=1))
    y_T = torch.transpose(y, 1, 2)
    for it in range(1, 501):
        out = pigno(ring_road.rho_init, ring_road.V_terminal)
        if it % 1000 == 0:
            out = pigno.mpnn_helper(out)
        loss = PINO_loss(
            ring_road, out, y_T, ic_weight=0.7, f_weight=0.3, data_weight=0.0
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"It: {it}, loss={float(loss)}")

    plot_3d(8, 8, out.detach().numpy()[0, :, :8], None)
    plot_3d(8, 8, out.detach().numpy()[0, :, 8:16], None)
    plot_3d(8, 8, y_T.detach().numpy()[0, :, :8], None)
    plot_3d(8, 8, y_T.detach().numpy()[0, :, 8:16], None)
