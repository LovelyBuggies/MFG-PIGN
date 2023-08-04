from argparse import ArgumentParser
import yaml
import numpy as np
import scipy.io
import torch
from src.loader import RingRoadLoader
from src.utils import get_args_kwargs
from src.runner import run_rho, run_V


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description="Basic paser")
    parser.add_argument(
        "--config_path", type=str, help="Path to the configuration file"
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as stream:
        config = yaml.load(stream, yaml.FullLoader)

    """Loader"""
    mat_file_path = config["data"]["file_path"]
    sample_num = config["data"]["sample_num"]
    check_id = config["data"]["check_id"]
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

    run_rho(ring_loader, args, config, check_id, test=True, show=True)
    run_V(ring_loader, args, config, check_id, test=True, show=True)
