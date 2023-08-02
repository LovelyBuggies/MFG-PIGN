import numpy as np
from loader import RingRoadLoader
from utils import plot_3d


def all_transition_tester(
    ring_loader, all_transitions, all_cumulative_transitions, wanna_check=0
):
    plot_3d(8, 8, ring_loader.rhos[wanna_check], "pre")
    rhos_1 = np.zeros(
        (ring_loader.n_samples, ring_loader.N, ring_loader.T), dtype=np.float32
    )
    for sample_i in range(ring_loader.n_samples):
        for t in range(ring_loader.T):
            rhos_1[sample_i, :, t] = np.dot(
                all_cumulative_transitions[sample_i][t],
                ring_loader.init_rhos[sample_i, :],
            )

    plot_3d(8, 8, rhos_1[wanna_check], "pre")

    rhos_2 = np.zeros(
        (ring_loader.n_samples, ring_loader.N, ring_loader.T), dtype=np.float32
    )
    for sample_i in range(ring_loader.n_samples):
        rhos_2[sample_i, :, 0] = ring_loader.init_rhos[sample_i, :]
        prev_rho_t = rhos_2[sample_i, :, 0]
        for t in range(1, ring_loader.T):
            rhos_2[sample_i, :, t] = np.dot(all_transitions[sample_i][t], prev_rho_t)
            prev_rho_t = rhos_2[sample_i, :, t]

    plot_3d(8, 8, rhos_2[wanna_check], "pre")
