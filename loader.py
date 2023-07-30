import numpy as np


class RingRoad:
    def __init__(self, rho, V):
        # rho and u are of shape (N, T)
        super().__init__()
        self.n_sample, self.N, self.T = rho.shape
        assert (V.shape[1], V.shape[2]) == (self.N + 1, self.T + 1)

        self.rho = rho
        self.V = V
        self.rho_init = self.rho[:, :, 0]
        self.V_terminal = self.V[:, :, -1]

        a = np.zeros((1, self.N))[0]
        b = np.ones((1, self.N - 1))[0]
        self.A = np.diag(a, 0) + np.diag(b, -1) + np.diag(b, 1)
        self.A[0, -1] = 1
        self.A[-1, 0] = 1
        self.A = np.repeat(self.A[None, :, :], self.n_sample, axis=0)
