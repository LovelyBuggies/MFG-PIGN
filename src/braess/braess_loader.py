import numpy as np
from src.utils import plot_4d


class BraessLoader:
    def __init__(self, rhos, us, Vs, betas, pis):
        super().__init__()
        self.N_nodes, self.N_edges = 4, 5
        self.origins, self.destinations = [0], [3]

        self.edges = np.zeros((5, 2), dtype=int)
        self.edges[0, 0] = 0
        self.edges[0, 1] = 1
        self.edges[1, 0] = 0
        self.edges[1, 1] = 2
        self.edges[2, 0] = 1
        self.edges[2, 1] = 3
        self.edges[3, 0] = 2
        self.edges[3, 1] = 3
        self.edges[4, 0] = 1
        self.edges[4, 1] = 2

        self.c = np.zeros((5, 1))
        self.c[0] = 1
        self.c[1] = 1.5
        self.c[2] = 1.5
        self.c[3] = 1
        self.c[4] = 1

        self.rhos, self.us, self.Vs = rhos, us, Vs
        self.betas, self.pis = betas, pis
        self.n_samples, _, self.N, self.T = rhos.shape
        self.delta_x, self.delta_t = 1 / self.N, 1 / self.N
        self.demands = np.zeros((1, self.T), dtype=np.float32)
        self.demands[:, 0] = 0.8

        self.init_rhos = np.concatenate(
            (rhos[:, :, :, 0], np.ones((self.n_samples, self.N_edges, 1))),
            axis=-1,
            dtype=np.float32,
        )
        self.terminal_Vs = np.ones(
            (self.n_samples, self.N_edges, self.N + 2), dtype=np.float32
        )
        self.terminal_Vs[:, :, :-2] = Vs[:, :, :, -1]

    def get_trans_matrix_rho(self, us, rhos, betas):
        n_samples, N_edges, N, T = us.shape
        delta_x, delta_t = 1 / N, 1 / N
        all_trans, all_cum_trans = list(), list()
        for sample_i in range(n_samples):
            trans = np.zeros((N_edges, N + 1, N + 1, T), dtype=np.float32)
            cum_trans = np.zeros((N_edges, N + 1, N + 1, T), dtype=np.float32)
            for edge_i in range(N_edges):
                trans[edge_i, :, :, 0] = np.eye(N + 1, dtype=np.float32)
                cum_trans[edge_i, :, :, 0] = np.eye(N + 1, dtype=np.float32)
                for t in range(1, T):
                    node_i = self.edges[edge_i, 0]
                    if node_i in self.origins:
                        demand_idx = self.origins.index(self.edges[edge_i, 0])
                        trans[edge_i, 0, -1, t] += (
                            (delta_t / delta_x)
                            * self.demands[demand_idx, t - 1]
                            * betas[sample_i, edge_i, t - 1]
                        )
                    if node_i in self.edges[:, 1]:
                        for e in range(self.N_edges):
                            if self.edges[e, 1] == node_i:
                                influx = (
                                    (delta_t / delta_x)
                                    * us[sample_i, e, -1, t - 1]
                                    * rhos[sample_i, e, -1, t - 1]
                                    * betas[sample_i, edge_i, t - 1]
                                )
                                trans[edge_i, 0, -1, t] += influx

                    for i in range(N):
                        trans[edge_i, i, i, t] = (
                            1 - (delta_t / delta_x) * us[sample_i, edge_i, i, t - 1]
                        )
                        if i != 0:
                            trans[edge_i, i, i - 1, t] = (delta_t / delta_x) * us[
                                sample_i, edge_i, i - 1, t - 1
                            ]

                    trans[edge_i, -1, -1, t] = 1
                    cum_trans[edge_i, :, :, t] = np.dot(
                        trans[edge_i, :, :, t], cum_trans[edge_i, :, :, t - 1]
                    )

            all_trans.append(trans)
            all_cum_trans.append(cum_trans)

        return all_trans, all_cum_trans

    def get_trans_matrix_V(self, us, rhos, pis):
        n_samples, N_edges, N, T = us.shape
        delta_x, delta_t = 1 / N, 1 / N
        all_trans, all_cum_trans = list(), list()
        for sample_i in range(n_samples):
            trans = np.zeros((N_edges, N + 2, N + 2, T + 1), dtype=np.float32)
            cum_trans = np.zeros((N_edges, N + 2, N + 2, T + 1), dtype=np.float32)
            for edge_i in range(N_edges):
                trans[edge_i, :, :, T] = np.eye(N + 2, dtype=np.float32)
                cum_trans[edge_i, :, :, T] = np.eye(N + 2, dtype=np.float32)
                for t in range(T - 1, -1, -1):
                    for i in range(N):
                        trans[edge_i, i, i, t] = (
                            1 - (delta_t / delta_x) * us[sample_i, edge_i, i, t]
                        )
                        if i < N - 1:
                            trans[edge_i, i, i + 1, t] = (delta_t / delta_x) * us[
                                sample_i, edge_i, i, t
                            ]
                        elif i == N - 1:
                            end_node = self.edges[edge_i, 1]
                            trans[edge_i, i, i + 1, t] = (
                                (delta_t / delta_x)
                                * us[sample_i, edge_i, i, t]
                                * pis[sample_i, end_node, t + 1]
                            )

                        r_i_t = delta_t * (
                            0.5 * us[sample_i, edge_i, i, t] ** 2
                            + rhos[sample_i, edge_i, i, t] * us[sample_i, edge_i, i, t]
                            - us[sample_i, edge_i, i, t]
                            + self.c[edge_i]
                        )
                        trans[edge_i, i, -1, t] = r_i_t

                    trans[edge_i, -2, -2, t] = 1
                    trans[edge_i, -1, -1, t] = 1
                    cum_trans[edge_i, :, :, t] = np.dot(
                        trans[edge_i, :, :, t], cum_trans[edge_i, :, :, t + 1]
                    )

            all_trans.append(trans)
            all_cum_trans.append(cum_trans)

        return all_trans, all_cum_trans
