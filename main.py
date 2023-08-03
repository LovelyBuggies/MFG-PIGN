import yaml
import numpy as np
import scipy.io
import torch
from src.loader import RingRoadLoader
from src.model import PIGN
from src.loss import supervised_loss, transition_loss
from src.utils import plot_3d, get_args_kwargs
from src.test import all_transition_tester


def runner(ring_loader, args, config, test=True):
    model = PIGN(*args)
    optimizer_kwargs = {"lr": config["train"]["lr"]}
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad is True], **optimizer_kwargs
    )
    loss_kwargs = {
        "func": torch.nn.MSELoss(),
        "w_ic": config["train"]["w_ic"],
        "w_physics": config["train"]["w_physics"],
    }

    """Params"""
    all_transitions, all_cum_transitions = ring_loader.get_transition_matrix()
    if test:
        all_transition_tester(
            ring_loader, all_transitions, all_cum_transitions, check_id
        )

    init_rho_copies = np.repeat(
        (ring_loader.init_rhos[:, :, None]), ring_loader.T, axis=-1
    )
    model_input = np.transpose(init_rho_copies, (0, 2, 1))
    messages = np.zeros(
        (ring_loader.n_samples, ring_loader.T, ring_loader.N, ring_loader.N, 1),
        dtype=np.float32,
    )
    for sample_i in range(ring_loader.n_samples):
        for t in range(ring_loader.T):
            messages[sample_i, t, :, :, 0] = all_cum_transitions[sample_i][t]

    """Train"""
    for it in range(config["train"]["epochs"]):
        # forward input as (T, X)
        model_output = model(model_input, messages=messages)
        preds = torch.transpose(model_output, 2, 1)
        loss = 0 * supervised_loss(
            preds,
            torch.from_numpy(rho_labels),
            loss_kwargs,
        ) + 1 * transition_loss(
            preds,
            all_transitions,
            ring_loader.init_rhos,
            loss_kwargs,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("it=", it, "loss=", float(loss))

    plot_3d(8, 8, ring_loader.rhos[check_id], f"truth-{check_id}")
    plot_3d(8, 8, preds[check_id].detach().numpy(), f"pred-{check_id}")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("config.ymal", "r") as stream:
        config = yaml.load(stream, yaml.FullLoader)

    mat_file_path = config["data"]["file_path"]
    sample_num = config["data"]["sample_num"]
    check_id = config["data"]["check_id"]

    """Loader"""
    rho_labels = scipy.io.loadmat(mat_file_path)["rhos"]
    u_labels = scipy.io.loadmat(mat_file_path)["us"]
    V_labels = scipy.io.loadmat(mat_file_path)["Vs"]
    rho_labels = np.array(rho_labels, dtype=np.float32)[:sample_num, :, :]
    u_labels = np.array(u_labels, dtype=np.float32)[:sample_num, :, :]
    V_labels = np.array(V_labels, dtype=np.float32)[:sample_num, :, :]
    ring_loader = RingRoadLoader(rho_labels, u_labels, V_labels)

    """Hyper-params"""
    f_channel_args, f_channel_kwargs = get_args_kwargs(
        config["model"]["f_channel"], device
    )
    f_sum_args, f_sum_kwargs = get_args_kwargs(config["model"]["f_sum"], device)
    f_x_args, f_x_kwargs = get_args_kwargs(config["model"]["f_x"], device)
    args = (
        f_channel_args,
        f_channel_kwargs,
        f_sum_args,
        f_sum_kwargs,
        f_x_args,
        f_x_kwargs,
    )

    runner(ring_loader, args, config, test=True)
