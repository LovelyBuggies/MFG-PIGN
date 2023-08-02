import numpy as np
from loader import RingRoadLoader
from utils import plot_3d

if __name__ == "__main__":
    rho_label = np.loadtxt("data/rho.csv", delimiter=",", dtype=np.float32)
    u_label = np.loadtxt("data/u.csv", delimiter=",", dtype=np.float32)
    V_label = np.loadtxt("data/V.csv", delimiter=",", dtype=np.float32)
    ring_data_loader = RingRoadLoader(rho_label, u_label, V_label)
    rho_inits = np.repeat(
        (ring_data_loader.init_rho[:, None]), ring_data_loader.T, axis=1
    )
    transitions, cumulative_transitions = ring_data_loader.get_transition_matrix()

    rho = np.zeros((ring_data_loader.N, ring_data_loader.T), dtype=np.float32)
    for t in range(ring_data_loader.T):
        rho[:, t] = np.dot(cumulative_transitions[t], rho_inits[:, t])
    plot_3d(8, 8, rho, "pre")

    rho = np.zeros((ring_data_loader.N, ring_data_loader.T), dtype=np.float32)
    rho[:, 0] = rho_inits[:, 0]
    prev_rho_t = rho[:, 0]
    for t in range(1, ring_data_loader.T):
        rho[:, t] = np.dot(transitions[t], prev_rho_t)
        prev_rho_t = rho[:, t]

    plot_3d(8, 8, rho, "pre")
