import matplotlib.pyplot as plt
import numpy as np


def nlms_filter(d, x, M, mu, epsilon=1e-3):
    n_samples = len(d)
    w = np.zeros(M)
    e = np.zeros(n_samples)

    for n in range(M, n_samples):
        x_n = x[n : n - M : -1]
        est_noise = np.dot(w, x_n)
        e[n] = d[n] - est_noise
        norm_x = np.dot(x_n, x_n)
        w = w + (mu / (norm_x + epsilon)) * e[n] * x_n

    return e


# 1. Signal Generation
fs = 8000
t = np.arange(fs * 3) / fs  # 3 seconds of data
M = 64  # Filter order
mu = 0.1  # Step size

# Clean Reference: 100Hz Sine Wave
clean = np.sin(2 * np.pi * 100 * t)

# Noise: Correlated noise with a 25-sample delay
ref_noise = np.random.normal(0, 1, len(t))
corrupted_noise = 0.7 * np.roll(ref_noise, 25)

# Primary Input: Clean + Noise
d = clean + corrupted_noise

# 2. Run NLMS
denoised = nlms_filter(d, ref_noise, M, mu)

# 3. Plotting (Showing only the last 0.05 seconds for clarity)
view_samples = 400

# Graph 1: Original Clean Signal
plt.figure(figsize=(10, 3))
plt.plot(t[-view_samples:], clean[-view_samples:], color="blue")
plt.title("1. Original Clean Signal")
plt.grid(True)
plt.ylim(-2, 2)

# Graph 2: Corrupted Signal (Input)
plt.figure(figsize=(10, 3))
plt.plot(t[-view_samples:], d[-view_samples:], color="red", alpha=0.7)
plt.title("2. Corrupted Signal (Clean + Noise)")
plt.grid(True)
plt.ylim(-4, 4)

# Graph 3: Denoised Signal (NLMS Output)
plt.figure(figsize=(10, 3))
plt.plot(t[-view_samples:], denoised[-view_samples:], color="green")
plt.title("3. Denoised Signal (NLMS Result)")
plt.grid(True)
plt.ylim(-2, 2)

plt.tight_layout()
plt.show()
