from argparse import ArgumentParser
import yaml
import numpy as np
import scipy.io
import torch
from src.utils import get_args_kwargs, plot_4d


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        run_rho_V(ring_loader, f_args, config, check_id, show=True)

    elif args.network == "braess":
        from src.braess.braess_loader import BraessLoader
        from src.braess.braess_tester import all_trans_tester_rho, all_trans_tester_V
        from src.braess.braess_runner import run_rho, run_V

        beta = scipy.io.loadmat(mat_file_path)["betas"].astype(np.float32)
        pi = scipy.io.loadmat(mat_file_path)["pis"].astype(np.float32)
        braess_loader = BraessLoader(rho, u, V, beta, pi)

        # all_trans, all_cum_trans = braess_loader.get_trans_matrix_rho(u, rho, beta)
        # all_trans_tester_rho(braess_loader, all_trans, all_cum_trans, 0)
        #
        # rho_message = np.repeat(
        #     (braess_loader.init_rhos[:, :, :-1, None]), braess_loader.T, axis=-1
        # )
        # for shuo in range(10):
        #     print(f"-------------------{shuo}-------------")
        #     rho_message, rho_loss = run_rho(
        #         braess_loader, u, rho_message, beta, f_args, config
        #     )
        #     plot_4d(
        #         braess_loader.N,
        #         int(braess_loader.T / braess_loader.N),
        #         rho_message[0],
        #         (0, 4, 3),
        #         "pred",
        #     )

        all_trans, all_cum_trans = braess_loader.get_trans_matrix_V(u, rho, pi)
        all_trans_tester_V(braess_loader, all_trans, all_cum_trans, 0)
        # V_message = np.repeat(
        #     (braess_loader.terminal_Vs[:, :, :-2, None]), braess_loader.T + 1, axis=-1
        # )
        # V_message = braess_loader.Vs
        # for shuo in range(1):
        #     print(f"-------------------{shuo}-------------")
        #     V_message, V_loss = run_V(braess_loader, u, rho, pi, f_args, config)
        #     plot_4d(
        #         braess_loader.N,
        #         int(braess_loader.T / braess_loader.N),
        #         V_message[0, :, :, :-1],
        #         (0, 4, 3),
        #         "pred",
        #     )
