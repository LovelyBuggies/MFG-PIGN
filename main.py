from argparse import ArgumentParser
import yaml
import numpy as np
import scipy.io
import torch
from src.utils import get_args_kwargs


if __name__ == "__main__":
    device = torch.device("cpu")
    parser = ArgumentParser(description="Basic parser")
    parser.add_argument("--network", type=str, help="The traffic network")
    parser.add_argument(
        "--config_path", type=str, help="Path to the configuration file"
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as stream:
        config = yaml.load(stream, yaml.FullLoader)

    mat_file_path = config["data"]["file_path"]
    check_id = config["data"]["check_id"]
    rho = scipy.io.loadmat(mat_file_path)["rhos"].astype(np.float32)
    u = scipy.io.loadmat(mat_file_path)["us"].astype(np.float32)
    V = scipy.io.loadmat(mat_file_path)["Vs"].astype(np.float32)

    f_channel_args, f_channel_kwargs = get_args_kwargs(config["model"]["f_channel"])
    f_sum_args, f_sum_kwargs = get_args_kwargs(config["model"]["f_sum"])
    f_x_args, f_x_kwargs = get_args_kwargs(config["model"]["f_x"])
    f_args = (
        f_channel_args,
        f_channel_kwargs,
        f_sum_args,
        f_sum_kwargs,
        f_x_args,
        f_x_kwargs,
    )
    if args.network == "ringroad":
        from src.ringroad.ringroad_loader import RingRoadLoader
        from src.ringroad.ringroad_runner import run_rho_V

        ring_loader = RingRoadLoader(rho, u, V)
        run_rho_V(ring_loader, f_args, config, check_id=check_id, show=True)

    elif args.network == "braess":
        from src.braess.braess_loader import BraessLoader
        from src.braess.braess_runner import run_rho_V

        beta = scipy.io.loadmat(mat_file_path)["betas"].astype(np.float32)
        pi = scipy.io.loadmat(mat_file_path)["pis"].astype(np.float32)
        demand = scipy.io.loadmat(mat_file_path)["demands"].astype(np.float32)
        terminal = scipy.io.loadmat(mat_file_path)["terminals"].astype(np.float32)
        braess_loader = BraessLoader(rho, u, V, beta, pi, demand, terminal)
        run_rho_V(braess_loader, f_args, config, check_id=check_id, show=True)
