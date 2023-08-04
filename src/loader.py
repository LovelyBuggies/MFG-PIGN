import numpy as np


class RingRoadLoader:
    def __init__(self, rhos, us, Vs):
        super().__init__()

        self.rhos, self.us, self.Vs = rhos.copy(), us.copy(), Vs.copy()
        self.n_samples, self.N, self.T = rhos.shape
        self.init_rhos, self.terminal_Vs = rhos[:, :, 0].copy(), Vs[:, :, -1].copy()
        self.terminal_Vs[:, -1] = 1  # reward dimension
        self.delta_x, self.delta_t = 1 / self.N, 1 / self.T

    def get_trans_matrix_rho(self):
        all_trans, all_cum_trans = list(), list()
        for sample_i in range(self.n_samples):
            trans, cum_trans = list(), list()
            P_prev = np.eye(self.N, dtype=np.float32)
            trans.append(P_prev)
            cum_trans.append(P_prev)
            for t in range(1, self.T):
                P = np.zeros((self.N, self.N), dtype=np.float32)
                for i in range(self.N):
                    P[i, i - 1] = (self.delta_t / self.delta_x) * self.us[
                        sample_i, i - 1, t - 1
                    ]
                    P[i, i] = (self.delta_t / self.delta_x) * (
                        1 - self.us[sample_i, i, t - 1]
                    )

                trans.append(P)
                P_prev = np.dot(P, P_prev)
                cum_trans.append(P_prev)

            all_trans.append(trans)
            all_cum_trans.append(cum_trans)

        return all_trans, all_cum_trans

    def get_trans_matrix_V(self):
        all_trans, all_cum_trans = list(), list()
        for sample_i in range(self.n_samples):
            trans, cum_trans = list(), list()
            P_prev = np.eye(self.N + 1, dtype=np.float32)
            trans.append(P_prev)
            cum_trans.append(P_prev)
            for t in range(self.T - 1, -1, -1):
                P = np.zeros((self.N + 1, self.N + 1), dtype=np.float32)
                for i in range(self.N):
                    P[i, i] = (self.delta_t / self.delta_x) * (
                        1 - self.us[sample_i, i, t]
                    )
                    if i < self.N - 1:
                        P[i, i + 1] = (self.delta_t / self.delta_x) * self.us[
                            sample_i, i, t
                        ]
                    else:
                        P[i, 0] = (self.delta_t / self.delta_x) * self.us[
                            sample_i, i, t
                        ]
                    r_i_t = self.delta_t * (
                        0.5 * self.us[sample_i, i, t] ** 2
                        + self.rhos[sample_i, i, t] * self.us[sample_i, i, t]
                        - self.us[sample_i, i, t]
                    )
                    P[i, -1] = r_i_t

                P[-1, -1] = 1  # to keep reward dimension
                trans = [P] + trans
                P_prev = np.dot(P, P_prev)  # may be changed
                cum_trans = [P_prev] + cum_trans

            all_trans.append(trans)
            all_cum_trans.append(cum_trans)

        return all_trans, all_cum_trans
