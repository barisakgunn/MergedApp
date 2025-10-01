import numpy as np
from numba import njit
import streamlit as st

# Parameters
n = 10                 # lattice size
kB = 0.0024          # Boltzmann constant
dT = 1e-2              # convergence threshold

# Initialize random spin lattice (-1 or +1)
lattice = np.random.choice([-1, 1], size=(n, n, n))

# Temperature bounds (avoid zero)
T_low = 1e-3
T_high = 100
T = [999, 100 * np.random.rand()]  # initial temps

# --- JIT-accelerated Monte Carlo step ---
@njit
def monte_carlo_step(lattice, J, T, kB, steps):
    n = lattice.shape[0]
    for _ in range(steps):
        x = np.random.randint(0, n)
        y = np.random.randint(0, n)
        z = np.random.randint(0, n)
        s = lattice[x, y, z]

        xa, xb = (x + 1) % n, (x - 1) % n
        ya, yb = (y + 1) % n, (y - 1) % n
        za, zb = (z + 1) % n, (z - 1) % n

        nearby_points = (
            lattice[xa, y, z] + lattice[xb, y, z] +
            lattice[x, ya, z] + lattice[x, yb, z] +
            lattice[x, y, za] + lattice[x, y, zb]
        )

        dE = J * s * nearby_points

        if dE <= 0 or np.random.rand() < np.exp(-dE / (T * kB)):
            lattice[x, y, z] = -s
    return lattice

# --- Main simulation ---
i = 1
while abs(T[i] - T[i - 1]) > dT:
    J = 1.0 / T[i]   # Ferromagnetic coupling

    # --- Warm-up sweeps (thermalization) ---
    lattice = monte_carlo_step(lattice, J, T[i], kB, steps=20_000)

    # --- Measurement sweeps ---
    lattice = monte_carlo_step(lattice, J, T[i], kB, steps=100_000)

    # Normalized magnetization per spin
    Ferromagneticism = abs(np.sum(lattice)) / (n**3)

    if Ferromagneticism > 0.9:
        T_low = T[i]
    else:
        T_high = T[i]

    T.append((T_low + T_high) / 2)
    i += 1

# --- Show results in Streamlit ---
st.title("Ising Model Cutoff Temperature Finder")
st.write(f"✅ Estimated Cutoff Temperature = **{T[-1]:.4f}**")
