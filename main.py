from argparse import ArgumentParser
import yaml
import numpy as np
import scipy.io
import torch
from src.loader import RingRoad
from src.utils import get_args_kwargs
from src.runner import run_rho, run_V, run_rho_V


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description="Basic paser")
    parser.add_argument(
        "--config_path", type=str, help="Path to the configuration file"
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as stream:
        config = yaml.load(stream, yaml.FullLoader)

    mat_file_path = config["data"]["file_path"]
    check_id = config["data"]["check_id"]
    rho_labels = scipy.io.loadmat(mat_file_path)["rhos"].astype(np.float32)
    u_labels = scipy.io.loadmat(mat_file_path)["us"].astype(np.float32)
    V_labels = scipy.io.loadmat(mat_file_path)["Vs"].astype(np.float32)
    ring_loader = RingRoad(rho_labels, u_labels, V_labels)

    f_channel_args, f_channel_kwargs = get_args_kwargs(config["model"]["f_channel"])
    f_sum_args, f_sum_kwargs = get_args_kwargs(config["model"]["f_sum"])
    f_x_args, f_x_kwargs = get_args_kwargs(config["model"]["f_x"])
    args = (
        f_channel_args,
        f_channel_kwargs,
        f_sum_args,
        f_sum_kwargs,
        f_x_args,
        f_x_kwargs,
    )

    run_rho_V(ring_loader, args, config, check_id, show=True)
