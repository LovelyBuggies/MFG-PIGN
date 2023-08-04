import numpy as np


class RingRoadLoader:
    def __init__(self, rhos, us, Vs):
        super().__init__()

        self.rhos, self.us, self.Vs = rhos, us, Vs
        self.n_samples, self.N, self.T = rhos.shape
        self.init_rhos, self.terminal_Vs = rhos[:, :, 0], Vs[:, :, -1]
        self.delta_x, self.delta_t = 1 / self.N, 1 / self.T

    def get_transition_matrix(self):
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
