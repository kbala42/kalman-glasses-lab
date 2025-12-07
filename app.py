import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Case 9 – Kalman’s Glasses Lab",
    page_icon="🕶️",
)

st.title("🕶️ Case 9: Kalman’s Glasses – Seeing Through Noise")
st.write(
    """
In this lab you put on **Kalman’s Glasses** for a spring–mass system.

- The **true motion** follows a known physical model (spring + damping).  
- The **sensor** gives a **noisy position measurement**.  
- The **Kalman filter** combines both stories to estimate position and velocity.

By changing the **process noise** and **measurement noise** levels you control
how much the filter trusts the model vs the sensor.
"""
)

st.markdown("---")

# -----------------------------
# 1) Physical model parameters
# -----------------------------
st.subheader("1️⃣ Spring–Mass System")

col_sys1, col_sys2, col_sys3 = st.columns(3)

with col_sys1:
    m = st.slider(
        "Mass m",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5,
        help="Heavier mass moves more slowly for the same force.",
    )
with col_sys2:
    k = st.slider(
        "Spring constant k",
        min_value=0.5,
        max_value=15.0,
        value=5.0,
        step=0.5,
        help="Larger k → stiffer spring, higher natural frequency.",
    )
with col_sys3:
    c = st.slider(
        "Damping c",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="Larger c → stronger friction / damping.",
    )

st.write(
    f"System: **m = {m:.1f}**, **k = {k:.1f}**, **c = {c:.1f}**"
)

# -----------------------------
# 2) Simulation settings
# -----------------------------
st.subheader("2️⃣ Simulation Timeline")

col_time1, col_time2 = st.columns(2)

with col_time1:
    t_max = st.slider(
        "Total time (s)",
        min_value=2.0,
        max_value=20.0,
        value=10.0,
        step=1.0,
    )
with col_time2:
    dt = st.slider(
        "Time step Δt (s)",
        min_value=0.01,
        max_value=0.1,
        value=0.02,
        step=0.01,
    )

n_steps = int(t_max / dt) + 1
st.write(
    f"Simulation: **{t_max:.1f} s**, Δt = **{dt:.3f} s**, steps ≈ **{n_steps}**"
)

col_init1, col_init2 = st.columns(2)
with col_init1:
    x0 = st.slider(
        "Initial position x₀",
        min_value=-2.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
    )
with col_init2:
    v0 = st.slider(
        "Initial velocity v₀",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
    )

# -----------------------------
# 3) Noise and Kalman settings
# -----------------------------
st.subheader("3️⃣ Noise & Kalman Filter Settings")

col_noise1, col_noise2, col_noise3 = st.columns(3)

with col_noise1:
    process_noise_level = st.slider(
        "Process noise level q",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
        help="Higher q = model is less trusted (more randomness in dynamics).",
    )
with col_noise2:
    measurement_noise_level = st.slider(
        "Measurement noise level r",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.02,
        help="Higher r = sensor is more noisy / less trusted.",
    )
with col_noise3:
    P0_scale = st.slider(
        "Initial covariance scale (P₀)",
        min_value=0.1,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="How uncertain the filter is about its initial guess.",
    )

col_guess1, col_guess2 = st.columns(2)
with col_guess1:
    xhat0 = st.slider(
        "Initial guess x̂₀",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
    )
with col_guess2:
    vhat0 = st.slider(
        "Initial guess v̂₀",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
    )

st.caption(
    "Kalman intuition: small q + large r → trust model more; "
    "large q + small r → trust measurements more."
)

# -----------------------------
# 4) Build discrete-time model (same as observer lab style)
# -----------------------------
st.subheader("4️⃣ Discrete-Time Model")

A11 = 1.0
A12 = dt
A21 = -(k / m) * dt
A22 = 1.0 - (c / m) * dt

A_d = np.array([[A11, A12],
                [A21, A22]])
C = np.array([[1.0, 0.0]])  # we measure position only

# Process and measurement noise covariances
Q = process_noise_level * np.eye(2)
R = np.array([[measurement_noise_level ** 2]])  # variance

A_tex = (
    r"A_d = \begin{{bmatrix}} {a11:.3f} & {a12:.3f} \\ {a21:.3f} & {a22:.3f} \end{{bmatrix}}"
    .format(a11=A11, a12=A12, a21=A21, a22=A22)
)
C_tex = r"C = \begin{bmatrix} 1 & 0 \end{bmatrix}"

st.latex(A_tex)
st.latex(C_tex)
st.caption(
    r"Model: $X_{k+1} = A_d X_k + w_k$,  $y_k = C X_k + v_k$."
)

# -----------------------------
# 5) Simulation + Kalman filter
# -----------------------------
def simulate_and_kalman(
    A_d, C, Q, R,
    x0, v0,
    xhat0, vhat0,
    dt, n_steps,
    noise_proc_scale,
    noise_meas_scale,
    seed=0,
):
    rng = np.random.default_rng(seed)

    t = np.zeros(n_steps)
    X_true = np.zeros((2, n_steps))
    y_meas = np.zeros(n_steps)

    X_hat = np.zeros((2, n_steps))
    P_hist = np.zeros((2, 2, n_steps))

    # Initial true state
    X_true[:, 0] = [x0, v0]

    # Initial estimate
    X_hat[:, 0] = [xhat0, vhat0]
    P = P0_scale * np.eye(2)
    P_hist[:, :, 0] = P

    for k in range(n_steps - 1):
        # ----- True system -----
        # Process noise (additive on state)
        w = noise_proc_scale * rng.standard_normal(size=2)
        X_true[:, k + 1] = A_d @ X_true[:, k] + w
        t[k + 1] = t[k] + dt

        # Measurement: position + noise
        v = noise_meas_scale * rng.standard_normal()
        y_meas[k] = C @ X_true[:, k] + v

        # ----- Kalman filter -----
        # Prediction
        X_pred = A_d @ X_hat[:, k]
        P_pred = A_d @ P @ A_d.T + Q

        # Update
        y_pred = C @ X_pred
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)

        innovation = y_meas[k] - y_pred[0]
        X_update = X_pred + (K[:, 0] * innovation)

        I = np.eye(2)
        P_update = (I - K @ C) @ P_pred

        X_hat[:, k + 1] = X_update
        P = P_update
        P_hist[:, :, k + 1] = P

    # Last measurement
    y_meas[-1] = C @ X_true[:, -1] + noise_meas_scale * rng.standard_normal()

    return t, X_true, y_meas, X_hat, P_hist


t, X_true, y_meas, X_hat, P_hist = simulate_and_kalman(
    A_d, C, Q, R,
    x0, v0,
    xhat0, vhat0,
    dt, n_steps,
    process_noise_level,
    measurement_noise_level,
)

x_true = X_true[0, :]
v_true = X_true[1, :]
x_hat = X_hat[0, :]
v_hat = X_hat[1, :]

# Standard deviations from covariance (for visual bands)
sigma_x = np.sqrt(P_hist[0, 0, :])
sigma_v = np.sqrt(P_hist[1, 1, :])

# -----------------------------
# 6) Plots
# -----------------------------
st.markdown("---")
st.subheader("5️⃣ Position – True vs Noisy vs Kalman Estimate")

fig1, ax1 = plt.subplots(figsize=(7, 4))

ax1.plot(t, x_true, label="True position x(t)")
ax1.plot(t, y_meas, linestyle=":", label="Noisy measurement y(t)")
ax1.plot(t, x_hat, label="Kalman estimate x̂(t)")

# Optional 1σ band
ax1.fill_between(
    t,
    x_hat - sigma_x,
    x_hat + sigma_x,
    alpha=0.2,
    label="±1σ (position uncertainty)",
)

ax1.set_xlabel("t (s)")
ax1.set_ylabel("Position")
ax1.set_title("Position vs Time")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

st.pyplot(fig1)

st.subheader("Velocity – True vs Kalman Estimate")

fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.plot(t, v_true, label="True velocity v(t)")
ax2.plot(t, v_hat, label="Kalman estimate v̂(t)")
ax2.set_xlabel("t (s)")
ax2.set_ylabel("Velocity")
ax2.set_title("Velocity vs Time")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

st.pyplot(fig2)

st.subheader("Estimation Errors")

e_x = x_true - x_hat
e_v = v_true - v_hat

fig3, ax3 = plt.subplots(figsize=(7, 3))
ax3.plot(t, e_x, label="Position error e_x = x − x̂")
ax3.plot(t, e_v, label="Velocity error e_v = v − v̂")
ax3.set_xlabel("t (s)")
ax3.set_ylabel("Error")
ax3.set_title("Kalman Estimation Errors")
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.legend()

st.pyplot(fig3)

# -----------------------------
# 7) Data table (first steps)
# -----------------------------
st.subheader("6️⃣ Sample Data Table (First 20 Steps)")

max_rows = min(20, n_steps)
df = pd.DataFrame(
    {
        "t (s)": t[:max_rows],
        "x_true": x_true[:max_rows],
        "y_meas": y_meas[:max_rows],
        "x_hat": x_hat[:max_rows],
        "v_true": v_true[:max_rows],
        "v_hat": v_hat[:max_rows],
        "σ_x": sigma_x[:max_rows],
        "σ_v": sigma_v[:max_rows],
    }
)

st.dataframe(
    df.style.format(
        {
            "t (s)": "{:.3f}",
            "x_true": "{:.3f}",
            "y_meas": "{:.3f}",
            "x_hat": "{:.3f}",
            "v_true": "{:.3f}",
            "v_hat": "{:.3f}",
            "σ_x": "{:.3f}",
            "σ_v": "{:.3f}",
        }
    )
)

# -----------------------------
# 8) Teacher / discussion box
# -----------------------------
st.markdown("---")
with st.expander("👩‍🏫 Teacher Box – Kalman Intuition & Questions"):
    st.write(
        r"""
**Key talking points**

- The filter combines two sources:
  - **Model prediction**: where we think the state will be if physics is correct.
  - **Measurement**: what the noisy sensor reports.
- The matrices **Q** and **R** control how much we trust each source.
- The covariance **P** tells us how unsure the filter still is; its diagonal terms
  (σ²) match the width of the uncertainty bands in the plots.

**Suggested questions**

1. Increase measurement noise (R) while keeping process noise small (Q).
   - How does the position estimate x̂(t) change?
   - Is it smoother or more jagged?
   - Does it lag behind the true position more or less?

2. Now do the opposite: keep R small, increase Q.
   - Does the estimate follow measurements more aggressively?
   - What happens to the uncertainty σ_x and σ_v?

3. Compare Kalman’s Glasses with the simple observer from Case 8:
   - In which situations might the extra complexity of a Kalman filter be worth it?
   - Where would a simpler observer be “good enough”?
"""
    )

st.caption(
    "Case 9 – Kalman’s Glasses: a gentle introduction to probabilistic state estimation "
    "for noisy spring–mass systems."
)
