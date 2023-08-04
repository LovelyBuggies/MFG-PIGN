import numpy as np


class RingRoadLoader:
    def __init__(self, rhos, us, Vs):
        super().__init__()

        self.rhos, self.us, self.Vs = rhos.copy(), us.copy(), Vs.copy()
        self.n_samples, self.N, self.T = rhos.shape
        self.init_rhos, self.terminal_Vs = rhos[:, :, 0].copy(), Vs[:, :, -1].copy()
        self.terminal_Vs[:, -1] = 1  # reward dimension
        self.delta_x, self.delta_t = 1 / self.N, 1 / self.T

    def get_u_from_rho_V(self, rhos, Vs):
        n_samples, N, T = rhos.shape
        delta_t, delta_x = 1 / N, 1 / T
        us = np.zeros((self.n_samples, self.N, self.T), dtype=np.float32)
        for sample_i in range(n_samples):
            for t in range(T):
                for i in range(N):
                    us[sample_i, i, t] = (
                        (Vs[sample_i, i, t + 1] - Vs[sample_i, i + 1, t + 1]) / delta_x
                        + 1
                        - rhos[sample_i, i, t]
                    )

        return us

    def get_trans_matrix_rho(self, us):
        n_samples, N, T = us.shape
        delta_t, delta_x = 1 / N, 1 / T
        all_trans, all_cum_trans = list(), list()
        for sample_i in range(n_samples):
            trans, cum_trans = list(), list()
            P_prev = np.eye(N, dtype=np.float32)
            trans.append(P_prev)
            cum_trans.append(P_prev)
            for t in range(1, T):
                P = np.zeros((N, N), dtype=np.float32)
                for i in range(N):
                    P[i, i - 1] = (delta_t / delta_x) * us[sample_i, i - 1, t - 1]
                    P[i, i] = (delta_t / delta_x) * (1 - us[sample_i, i, t - 1])

                trans.append(P)
                P_prev = np.dot(P, P_prev)
                cum_trans.append(P_prev)

            all_trans.append(trans)
            all_cum_trans.append(cum_trans)

        return all_trans, all_cum_trans

    def get_trans_matrix_V(self, us, rhos):
        n_samples, N, T = us.shape
        delta_t, delta_x = 1 / N, 1 / T
        all_trans, all_cum_trans = list(), list()
        for sample_i in range(n_samples):
            trans, cum_trans = list(), list()
            P_prev = np.eye(N + 1, dtype=np.float32)
            trans.append(P_prev)
            cum_trans.append(P_prev)
            for t in range(T - 1, -1, -1):
                P = np.zeros((N + 1, N + 1), dtype=np.float32)
                for i in range(N):
                    P[i, i] = (delta_t / delta_x) * (1 - us[sample_i, i, t])
                    if i < N - 1:
                        P[i, i + 1] = (delta_t / delta_x) * us[sample_i, i, t]
                    else:
                        P[i, 0] = (delta_t / delta_x) * us[sample_i, i, t]
                    r_i_t = delta_t * (
                        0.5 * us[sample_i, i, t] ** 2
                        + rhos[sample_i, i, t] * us[sample_i, i, t]
                        - us[sample_i, i, t]
                    )
                    P[i, -1] = r_i_t

                P[-1, -1] = 1  # to keep reward dimension
                trans = [P] + trans
                P_prev = np.dot(P, P_prev)  # may be changed
                cum_trans = [P_prev] + cum_trans

            all_trans.append(trans)
            all_cum_trans.append(cum_trans)

        return all_trans, all_cum_trans
