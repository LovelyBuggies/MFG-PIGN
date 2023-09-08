import numpy as np
from src.utils import plot_4d


def all_trans_tester_rho(braess_loader, all_trans, all_cum_trans, check_id=0):
    rhos_1 = np.zeros(
        (
            braess_loader.n_samples,
            braess_loader.N_edges,
            braess_loader.N + 1,
            braess_loader.T,
        ),
        dtype=np.float32,
    )
    for sample_i in range(braess_loader.n_samples):
        for edge_i in range(braess_loader.N_edges):
            for t in range(braess_loader.T):
                rhos_1[sample_i, edge_i, :, t] = np.dot(
                    all_cum_trans[sample_i][edge_i, :, :, t],
                    braess_loader.init_rhos[sample_i, edge_i, :],
                )

    rhos_2 = np.zeros(
        (
            braess_loader.n_samples,
            braess_loader.N_edges,
            braess_loader.N + 1,
            braess_loader.T,
        ),
        dtype=np.float32,
    )
    for sample_i in range(braess_loader.n_samples):
        for edge_i in range(braess_loader.N_edges):
            rhos_2[sample_i, edge_i, :, 0] = braess_loader.init_rhos[
                sample_i, edge_i, :
            ]
            prev_rho_t = rhos_2[sample_i, edge_i, :, 0]
            for t in range(1, braess_loader.T):
                rhos_2[sample_i, edge_i, :, t] = np.dot(
                    all_trans[sample_i][edge_i, :, :, t], prev_rho_t
                )
                prev_rho_t = rhos_2[sample_i, edge_i, :, t]

    plot_4d(
        braess_loader.N,
        int(braess_loader.T / braess_loader.N),
        rhos_1[check_id, :, :-1, :],
        (0, 4, 3),
        "cum_trans",
    )

    plot_4d(
        braess_loader.N,
        int(braess_loader.T / braess_loader.N),
        rhos_2[check_id, :, :-1, :],
        (0, 4, 3),
        "trans",
    )


def all_trans_tester_V(braess_loader, all_trans, all_cum_trans, check_id=0):
    Vs_1 = np.zeros(
        (
            braess_loader.n_samples,
            braess_loader.N_edges,
            braess_loader.N + 2,
            braess_loader.T + 1,
        ),
        dtype=np.float32,
    )
    for sample_i in range(braess_loader.n_samples):
        for edge_i in range(braess_loader.N_edges):
            for t in range(braess_loader.T + 1):
                Vs_1[sample_i, edge_i, :, t] = np.dot(
                    all_cum_trans[sample_i][edge_i, :, :, t],
                    braess_loader.terminal_Vs[sample_i, edge_i, :],
                )

    Vs_2 = np.zeros(
        (
            braess_loader.n_samples,
            braess_loader.N_edges,
            braess_loader.N + 2,
            braess_loader.T + 1,
        ),
        dtype=np.float32,
    )
    for sample_i in range(braess_loader.n_samples):
        for edge_i in range(braess_loader.N_edges):
            Vs_2[sample_i, edge_i, :, -1] = braess_loader.terminal_Vs[
                sample_i, edge_i, :
            ]
            curr_V_t = Vs_2[sample_i, edge_i, :, -1]
            for t in range(braess_loader.T, -1, -1):
                Vs_2[sample_i, edge_i, :, t] = np.dot(
                    all_trans[sample_i][edge_i, :, :, t], curr_V_t
                )
                curr_V_t = Vs_2[sample_i, edge_i, :, t]

    plot_4d(
        braess_loader.N,
        int(braess_loader.T / braess_loader.N),
        Vs_1[check_id, :, :-2, :-1],
        (0, 4, 3),
        "cum_trans",
    )

    plot_4d(
        braess_loader.N,
        int(braess_loader.T / braess_loader.N),
        Vs_2[check_id, :, :-2, :-1],
        (0, 4, 3),
        "trans",
    )
