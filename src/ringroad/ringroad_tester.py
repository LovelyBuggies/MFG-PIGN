import numpy as np
from src.utils import plot_3d


def all_trans_tester_rho(ring_loader, all_trans, all_cum_trans, check_id=0):
    rhos_1 = np.zeros(
        (ring_loader.n_samples, ring_loader.N, ring_loader.T), dtype=np.float32
    )
    for sample_i in range(ring_loader.n_samples):
        for t in range(ring_loader.T):
            rhos_1[sample_i, :, t] = np.dot(
                all_cum_trans[sample_i][t],
                ring_loader.init_rhos[sample_i, :],
            )

    plot_3d(ring_loader.N, ring_loader.T, rhos_1[check_id], "cum-trans")

    rhos_2 = np.zeros(
        (ring_loader.n_samples, ring_loader.N, ring_loader.T), dtype=np.float32
    )
    for sample_i in range(ring_loader.n_samples):
        rhos_2[sample_i, :, 0] = ring_loader.init_rhos[sample_i, :]
        prev_rho_t = rhos_2[sample_i, :, 0]
        for t in range(1, ring_loader.T):
            rhos_2[sample_i, :, t] = np.dot(all_trans[sample_i][t], prev_rho_t)
            prev_rho_t = rhos_2[sample_i, :, t]

    plot_3d(ring_loader.N, ring_loader.T, rhos_2[check_id], "trans")


def all_trans_tester_V(ring_loader, all_trans, all_cum_trans, check_id=0):
    Vs_1 = np.zeros(
        (ring_loader.n_samples, ring_loader.N + 1, ring_loader.T + 1), dtype=np.float32
    )
    for sample_i in range(ring_loader.n_samples):
        for t in range(ring_loader.T + 1):
            Vs_1[sample_i, :, t] = np.dot(
                all_cum_trans[sample_i][t],
                ring_loader.terminal_Vs[sample_i, :],
            )

    Vs_1[:, -1, :] = Vs_1[:, 0, :]
    plot_3d(ring_loader.N + 1, ring_loader.T + 1, Vs_1[check_id], "cum-trans")

    Vs_2 = np.zeros(
        (ring_loader.n_samples, ring_loader.N + 1, ring_loader.T + 1), dtype=np.float32
    )
    for sample_i in range(ring_loader.n_samples):
        Vs_2[sample_i, :, -1] = ring_loader.terminal_Vs[sample_i, :]
        curr_V_t = Vs_2[sample_i, :, -1]
        for t in range(ring_loader.T - 1, -1, -1):
            Vs_2[sample_i, :, t] = np.dot(all_trans[sample_i][t], curr_V_t)
            curr_V_t = Vs_2[sample_i, :, t]

    Vs_2[:, -1, :] = Vs_2[:, 0, :]
    plot_3d(ring_loader.N + 1, ring_loader.T + 1, Vs_2[check_id], "trans")
