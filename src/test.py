import numpy as np
from .utils import plot_3d


def all_trans_tester(ring_loader, all_trans, all_cum_trans, check_id=0):
    rhos_1 = np.zeros(
        (ring_loader.n_samples, ring_loader.N, ring_loader.T), dtype=np.float32
    )
    for sample_i in range(ring_loader.n_samples):
        for t in range(ring_loader.T):
            rhos_1[sample_i, :, t] = np.dot(
                all_cum_trans[sample_i][t],
                ring_loader.init_rhos[sample_i, :],
            )

    plot_3d(8, 8, rhos_1[check_id], "cum-trans")

    rhos_2 = np.zeros(
        (ring_loader.n_samples, ring_loader.N, ring_loader.T), dtype=np.float32
    )
    for sample_i in range(ring_loader.n_samples):
        rhos_2[sample_i, :, 0] = ring_loader.init_rhos[sample_i, :]
        prev_rho_t = rhos_2[sample_i, :, 0]
        for t in range(1, ring_loader.T):
            rhos_2[sample_i, :, t] = np.dot(all_trans[sample_i][t], prev_rho_t)
            prev_rho_t = rhos_2[sample_i, :, t]

    plot_3d(8, 8, rhos_2[check_id], "trans")
