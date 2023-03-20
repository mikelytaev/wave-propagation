import numpy as np

from utils import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


k_z_bounds, err = err_2d(
    freq_hz=300e6,
    dx_m=3.69,
    dz_m=0.076,
    theta_max_degrees=30,
    pade_order=(6, 7),
    shift=False,
    adi=False,
    n=500
)

k_z_bounds, err_adi = err_2d(
    freq_hz=300e6,
    dx_m=3.69,
    dz_m=0.076,
    theta_max_degrees=30,
    pade_order=(6, 7),
    shift=False,
    adi=True,
    n=500
)

k_z_bounds, err_shift = err_2d(
    freq_hz=300e6,
    dx_m=3.69,
    dz_m=0.076,
    theta_max_degrees=30,
    pade_order=(6, 7),
    shift=True,
    adi=False,
    n=500
)

k_z_bounds, err_shift_adi = err_2d(
    freq_hz=300e6,
    dx_m=3.69,
    dz_m=0.076,
    theta_max_degrees=30,
    pade_order=(6, 7),
    shift=True,
    adi=True,
    n=500
)

f, ax = plt.subplots(2, 2, figsize=(6, 2*3.2), constrained_layout=True)
extent = [k_z_bounds[0], k_z_bounds[1], k_z_bounds[0], k_z_bounds[1]]
norm = Normalize(-0, -70)
ax[0,0].imshow(10*np.log10(err), extent=extent, norm=norm, cmap=plt.get_cmap('binary'))
ax[0,0].set_title("3D CN")
ax[0,0].set_ylabel("k_y")
ax[0,1].imshow(10*np.log10(err_adi), extent=extent, norm=norm, cmap=plt.get_cmap('binary'))
ax[0,1].set_title("ADI")
ax[1,0].imshow(10*np.log10(err_shift), extent=extent, norm=norm, cmap=plt.get_cmap('binary'))
ax[1,0].set_title("3D CN, Shifted")
ax[1,0].set_ylabel("k_y")
ax[1,0].set_xlabel("k_z")
im = ax[1,1].imshow(10*np.log10(err_shift_adi), extent=extent, norm=norm, cmap=plt.get_cmap('binary'))
ax[1,1].set_title("ADI, Shifted")
ax[1,1].set_xlabel("k_z")
#plt.colorbar(fraction=0.046, pad=0.04)
f.colorbar(im, ax=ax[:], shrink=0.6, location='bottom')
plt.show()