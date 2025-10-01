import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Parameters
n = 5                 # lattice size
kB = 0.007             # Boltzmann constant

# Initialize random spin lattice (-1 or +1)
lattice = np.random.choice([-1, 1], size=(n, n, n))

# Temperature input (ensure T > 0)
T = st.number_input("Choose an arbitrary positive Temperature value:", min_value=1e-6, value=1.0)
J = 1.0 / T

# Monte Carlo simulation
for _ in range(1_000_000):
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

# --- 3D Scatter Plot ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Collect coordinates for spins
xs, ys, zs = np.indices((n, n, n))
xs, ys, zs = xs.flatten(), ys.flatten(), zs.flatten()
spins = lattice.flatten()

# Plot spins: red ^ for +1, blue v for -1
for x, y, z, s in zip(xs, ys, zs, spins):
    if s == 1:
        ax.scatter(x, y, z, c="red", marker="^", s=50)
    else:
        ax.scatter(x, y, z, c="blue", marker="v", s=50)

ax.set_title("3D Spin Glass Graph")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
st.pyplot(fig)
