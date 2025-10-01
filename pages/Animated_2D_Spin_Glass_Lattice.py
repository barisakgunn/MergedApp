import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time

# Parameters
n = 10
kB = 0.007
steps_per_frame = 500

# Streamlit UI
st.title("2D Ising Model Animation")
T = st.slider("Choose Temperature", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
J = 1.0 / T

# Initialize lattice in session state
if "lattice" not in st.session_state:
    st.session_state.lattice = np.random.choice([-1, 1], size=(n, n))
if "running" not in st.session_state:
    st.session_state.running = False

# Monte Carlo step
def monte_carlo_step(lattice, T, J, kB, steps=1):
    n = lattice.shape[0]
    for _ in range(steps):
        x, y = np.random.randint(0, n, size=2)
        s = lattice[x, y]

        xa, xb = (x + 1) % n, (x - 1) % n
        ya, yb = (y + 1) % n, (y - 1) % n

        neighbors_sum = lattice[xa, y] + lattice[xb, y] + lattice[x, ya] + lattice[x, yb]
        dE = 2 * J * s * neighbors_sum

        if dE <= 0 or np.random.rand() < np.exp(-dE / (kB * T)):
            lattice[x, y] = -s
    return lattice

# Control buttons
col1, col2, col3 = st.columns(3)
if col1.button("▶ Start Animation"):
    st.session_state.running = True
if col2.button("⏹ Stop Animation"):
    st.session_state.running = False
if col3.button("🔄 Reset Lattice"):
    st.session_state.lattice = np.random.choice([-1, 1], size=(n, n))
    st.session_state.running = False

# Placeholder for plot
plot_area = st.empty()

# Animation loop
if st.session_state.running:
    for frame in range(200):
        st.session_state.lattice = monte_carlo_step(
            st.session_state.lattice, T, J, kB, steps=steps_per_frame
        )

        fig, ax = plt.subplots()
        xs, ys = np.indices((n, n))
        xs, ys = xs.flatten(), ys.flatten()
        spins = st.session_state.lattice.flatten()

        colors = ["red" if s == 1 else "blue" for s in spins]
        markers = ["^" if s == 1 else "v" for s in spins]

        for x, y, s, c, m in zip(xs, ys, spins, colors, markers):
            ax.scatter(x, y, c=c, marker=m, s=100)

        ax.set_xlim(-1, n)
        ax.set_ylim(-1, n)
        ax.set_aspect("equal")
        ax.set_title(f"2D Ising Model (Frame {frame})")

        plot_area.pyplot(fig)
        plt.close(fig)

        time.sleep(0.2)

        if not st.session_state.running:  # stop if button pressed
            break
