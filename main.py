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

    rho_label = np.loadtxt("data/rho.csv", delimiter=",")
    u_label = np.loadtxt("data/u.csv", delimiter=",")
    V_label = np.loadtxt("data/V.csv", delimiter=",")
    ring_data_loader = RingRoadLoader(rho_label, u_label, V_label)
    rho_inits = np.repeat(
        (ring_data_loader.init_rho[:, None]), ring_data_loader.T, axis=1
    )
    V_terminals = np.repeat(
        (ring_data_loader.terminal_V[:, None]), ring_data_loader.T + 1, axis=1
    )

    rho = np.zeros(rho_label.shape, dtype=np.float32)
    transitions = ring_data_loader.get_transition_matrix()
    rho[:, 0] = ring_data_loader.init_rho
    prev_rho_t = rho[:, 0]
    # rho[:, 1] = ring_data_loader.rho[:, 1]
    # prev_rho_t = rho[:, 1]
    for t in range(1, ring_data_loader.T):
        curr_rho_t = np.dot(transitions[t - 1], prev_rho_t)
        rho[:, t] = curr_rho_t
        prev_rho_t = curr_rho_t

    plot_3d(8, 8, rho, "pre")
