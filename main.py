import numpy as np
import torch
from loader import RingRoadLoader
from model import MPNN
from utils import plot_3d


if __name__ == "__main__":
    INPUT_DIM = 2
    OUTPUT_DIM = 2
    N_HIDDEN = 4
    HIDDEN_DIM = 32
    early_stop = 500

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        DEVICE = torch.device("cpu")

    rho = np.loadtxt("data/rho.csv", delimiter=",")
    V = np.loadtxt("data/V.csv", delimiter=",")[:8, :8]
    ring_data_loader = RingRoadLoader(rho, V)
    X_train, y_train, X_test, y_test = ring_data_loader.train_test_split(ratio=0.7)

    f_x_args = (INPUT_DIM, OUTPUT_DIM, N_HIDDEN, HIDDEN_DIM)
    f_x_kwargs = {
        "activation_type": "none",
        "last_activation_type": "none",
        "device": DEVICE,
    }
    x = np.hstack([rho.reshape(-1, 1), V.reshape(-1, 1)])
    f_x_kwargs["mean"] = np.array([x.mean(0)[0], 0], dtype=np.float32)
    f_x_kwargs["std"] = np.array([x.std(0)[0], 1], dtype=np.float32)

    f_m_args = (INPUT_DIM, 1, N_HIDDEN, HIDDEN_DIM)
    f_m_kwargs = {
        "activation_type": "none",
        "last_activation_type": "none",
        "device": DEVICE,
        "mean": np.array([x.mean(0)[0], x.mean(0)[1]], dtype=np.float32),
        "std": np.array([x.std(0)[0], x.std(0)[1]], dtype=np.float32),
    }

    mpnn = MPNN(f_x_args, f_x_kwargs, f_m_args, f_m_kwargs, ring_data_loader.A)
    optimizer_kwargs = {"lr": 0.001}
    optimizer = torch.optim.Adam(
        [p for p in mpnn.parameters() if p.requires_grad is True], **optimizer_kwargs
    )
    loss_fun = torch.nn.MSELoss()

    # train
    best_mse, best_it, best_pred, best_val = 100000, 0, None, None
    for it in range(2000):
        pred = torch.squeeze(mpnn(X_train))
        loss = loss_fun(pred, torch.Tensor(y_train))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        val = torch.squeeze(mpnn(X_test))
        loss_val = float(loss_fun(val, torch.Tensor(y_test)))

        print(
            "it=",
            it,
            "loss_train=",
            float(loss),
            " loss_val=",
            loss_val,
        )
        if loss_val < best_mse:
            best_mse, best_it = loss_val, it
            best_pred, best_val = pred, val
        if it - best_it > early_stop:
            break

    # results
    out = np.concatenate(
        (
            X_train[:1, :, :],
            best_pred.cpu().detach().numpy(),
            X_test[:1, :, :],
            best_val.cpu().detach().numpy(),
        ),
        axis=0,
    )
    label = np.concatenate(
        (
            X_train[:1, :, :],
            y_train,
            X_test[:1, :, :],
            y_test,
        ),
        axis=0,
    )

    plot_3d(8, 8, out[:, :, 0], "pre")
    plot_3d(8, 8, label[:, :, 0], "pre")
    plot_3d(8, 8, out[:, :, 1], "pre")
    plot_3d(8, 8, label[:, :, 1], "pre")
