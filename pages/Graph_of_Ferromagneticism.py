import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- Parameters ---
n = 10                 # lattice size
kB = 0.1            # Boltzmann constant (chosen so Tc ~ 25)
minT = 4.0          # avoid T = 0
maxT = 40
stepT = 2.0
sample_N = int((maxT - minT) / stepT) + 1
mc_steps = 50000     # Monte Carlo steps per T
equil_steps = 10_000  # extra sweeps to equilibrate

# Streamlit UI
st.title("3D Ising Model Simulation")
st.write("Monte Carlo simulation of a 3D Ising model with temperature sweep.")

# Initialize random spin lattice (-1 or +1)
lattice = np.random.choice([-1, 1], size=(n, n, n))

# Store magnetization
Ferromagneticism = []

# --- Simulation ---
for T_loop in range(sample_N):
    T = minT + T_loop * stepT
    J = 1.0  # keep J constant!

    # --- Thermalize before measurement ---
    for _ in range(equil_steps):
        x, y, z = np.random.randint(0, n, size=3)
        s = lattice[x, y, z]
        xa, xb = (x + 1) % n, (x - 1) % n
        ya, yb = (y + 1) % n, (y - 1) % n
        za, zb = (z + 1) % n, (z - 1) % n
        neighbors_sum = (
            lattice[xa, y, z] + lattice[xb, y, z] +
            lattice[x, ya, z] + lattice[x, yb, z] +
            lattice[x, y, za] + lattice[x, y, zb]
        )
        dE = 2 * J * s * neighbors_sum
        if dE <= 0 or np.random.rand() < np.exp(-dE / (kB * T)):
            lattice[x, y, z] = -s

    # --- Measurement phase ---
    for _ in range(mc_steps):
        x, y, z = np.random.randint(0, n, size=3)
        s = lattice[x, y, z]
        xa, xb = (x + 1) % n, (x - 1) % n
        ya, yb = (y + 1) % n, (y - 1) % n
        za, zb = (z + 1) % n, (z - 1) % n
        neighbors_sum = (
            lattice[xa, y, z] + lattice[xb, y, z] +
            lattice[x, ya, z] + lattice[x, yb, z] +
            lattice[x, y, za] + lattice[x, y, zb]
        )
        dE = 2 * J * s * neighbors_sum
        if dE <= 0 or np.random.rand() < np.exp(-dE / (kB * T)):
            lattice[x, y, z] = -s

    # Magnetization (absolute sum of spins per site)
    Ferromagneticism.append(abs(np.sum(lattice)) / (n**3))

# --- Plot ---
fig, ax = plt.subplots()
T_plot = np.linspace(minT, maxT, sample_N)
ax.plot(T_plot, Ferromagneticism, ".", markersize=6)
ax.set_xlabel("Temperature (T)")
ax.set_ylabel("Ferromagneticism |M|")
ax.set_title("Temperature dependent Ferromagneticism")

st.pyplot(fig)
