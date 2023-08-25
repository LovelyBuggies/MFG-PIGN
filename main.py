from argparse import ArgumentParser
import yaml
import numpy as np
import scipy.io
import torch
from src.braess.braess_loader import BraessLoader
from src.braess.braess_tester import all_trans_tester_rho


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
    rho = scipy.io.loadmat(mat_file_path)["rhos"].astype(np.float32)
    u = scipy.io.loadmat(mat_file_path)["us"].astype(np.float32)
    V = scipy.io.loadmat(mat_file_path)["Vs"].astype(np.float32)
    beta = scipy.io.loadmat(mat_file_path)["betas"].astype(np.float32)
    braess_loader = BraessLoader(rho, u, V, beta)

    # to test
    all_trans = braess_loader.get_trans_matrix_rho(u, rho, beta)
    all_trans_tester_rho(braess_loader, all_trans, None)
