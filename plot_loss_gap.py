import numpy as np
import pandas as pd
import scipy
from src.braess.braess_loader import BraessLoader
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


mat_file_path = "data/braess/braess-config.mat"
rho = scipy.io.loadmat(mat_file_path)["rhos"].astype(np.float32)
u = scipy.io.loadmat(mat_file_path)["us"].astype(np.float32)
V = scipy.io.loadmat(mat_file_path)["Vs"].astype(np.float32)
beta = scipy.io.loadmat(mat_file_path)["betas"].astype(np.float32)
pi = scipy.io.loadmat(mat_file_path)["pis"].astype(np.float32)
demand = scipy.io.loadmat(mat_file_path)["demands"].astype(np.float32)
terminal = scipy.io.loadmat(mat_file_path)["terminals"].astype(np.float32)
braess_loader = BraessLoader(rho, u, V, beta, pi, demand, terminal)

rho_gap, u_gap = list(), list()
rho_loss, u_loss = list(), list()
rho_prev = np.repeat(
    (braess_loader.init_rhos[:, :, :-1, None]), braess_loader.T, axis=-1
)
V_prev = np.repeat(
    (braess_loader.terminal_Vs[:, :, :-2, None]), braess_loader.T + 1, axis=-1
)
_, pi_prev = braess_loader.get_beta_pi_from_V(V_prev)
u_prev = braess_loader.get_u_from_rho_V_pi(rho_prev, V_prev, pi_prev)
rho_loss.append(np.mean(abs(rho - rho_prev)))
u_loss.append(np.mean(abs(u - u_prev)))
for ep in range(0, 50):
    rho_curr = np.load(f"result/braess/rho-preds-{ep}.npy")
    u_curr = np.load(f"result/braess/u-preds-{ep}.npy")
    rho_gap.append(np.mean(abs(rho_curr - rho_prev)))
    u_gap.append(np.mean(abs(u_curr - u_prev)))
    rho_loss.append(np.mean(abs(rho - rho_curr)))
    u_loss.append(np.mean(abs(u - u_curr)))
    rho_prev, u_prev = rho_curr, u_curr


# plot_gap.py
fig, ax = plt.subplots(figsize=(10, 6))
rho_gap = savgol_filter([rho for rho in rho_gap], 8, 3)
u_gap = savgol_filter([u for u in u_gap], 8, 3)
plt.plot(
    rho_gap,
    lw=3,
    label=r"$|\rho^{(i)} - \rho^{(i-1)}|$",
    c="indianred",
    alpha=0.8,
)
plt.plot(u_gap, lw=3, label=r"$|u^{(i)} - u^{(i-1)}|$", c="steelblue", ls="--")
plt.xlabel("iterations", fontsize=18, labelpad=6)
plt.xticks(fontsize=18)
plt.ylabel("convergence gap", fontsize=18, labelpad=6)
plt.yticks(fontsize=18)
plt.legend(prop={"size": 16})
plt.savefig("pigno-gap.pdf")

# plot_loss.py
fig, ax = plt.subplots(figsize=(10, 6))
rho_loss = savgol_filter([rho for rho in rho_loss], 8, 3)
u_loss = savgol_filter([u for u in u_loss], 8, 3)
plt.plot(
    rho_loss,
    lw=3,
    label=r"$|\rho^{(i)} - \rho^*|$",
    c="indianred",
    alpha=0.8,
)
plt.plot(u_loss, lw=3, label=r"$|u^{(i)} - u^*|$", c="steelblue", ls="--")
plt.xlabel("iterations", fontsize=18, labelpad=6)
plt.xticks(fontsize=18)
plt.ylabel("W1-distance", fontsize=18, labelpad=6)
plt.yticks(fontsize=18)
plt.legend(prop={"size": 16})
plt.savefig("pigno-w1.pdf")
