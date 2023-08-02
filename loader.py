import numpy as np


class RingRoadLoader:
    def __init__(self, rho, u, V):
        super().__init__()

        self.rho, self.u, self.V = rho, u, V
        self.N, self.T = rho.shape
        self.init_rho, self.terminal_V = rho[:, 0], V[:, -1]
        self.delta_x, self.delta_t = 1 / self.N, 1 / self.T

        self.A = np.eye(self.N, dtype=np.float32)
        for i in range(self.N):
            self.A[i, i - 1] = 1.0

    def get_transition_matrix(self):
        transitions, cumulative_transitions = list(), list()
        P_prev = np.eye(self.N, dtype=np.float32)
        transitions.append(P_prev)
        cumulative_transitions.append(P_prev)
        for t in range(self.T):
            P = np.zeros((self.N, self.N), dtype=np.float32)
            for i in range(self.N):
                P[i, i - 1] = (self.delta_t / self.delta_x) * self.u[i - 1, t - 1]
                P[i, i] = (self.delta_t / self.delta_x) * (1 - self.u[i, t - 1])

            transitions.append(P)
            P_prev = np.dot(P, P_prev)
            cumulative_transitions.append(P_prev)

        assert len(transitions) == self.T + 1
        return transitions, cumulative_transitions
