import numpy as np

from utils import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


theta_max_degrees = 30

theta_grid_degrees, err = err_theta(
    freq_hz=300e6,
    dx_m=3.69*1.95,
    dz_m=0.076,
    theta_max_degrees=theta_max_degrees,
    phi_degrees=10,
    pade_order=(6, 7),
    shift=False,
    adi=False,
    n=500,
    z_order=4
)

theta_grid_degrees, err_shift = err_theta(
    freq_hz=300e6,
    dx_m=3.69*1.95,
    dz_m=0.076,
    theta_max_degrees=theta_max_degrees,
    phi_degrees=10,
    pade_order=(6, 7),
    shift=True,
    adi=False,
    n=500,
    z_order=4
)


f, ax = plt.subplots(1, 1, figsize=(6, 3.2), constrained_layout=True)
ax.plot(theta_grid_degrees, 10*np.log10(err), theta_grid_degrees, 10*np.log10(err_shift))
ax.grid(True)
ax.set_xlim([0, theta_max_degrees])
ax.set_ylim([-100, 0])
ax.set_ylabel("k_x error")
ax.set_xlabel("theta (degrees)")
plt.show()