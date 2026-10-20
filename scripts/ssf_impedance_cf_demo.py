"""Angle-dependent (consistent-field) vs mixed-Fourier (DMFT) impedance
boundary condition in the JAX split-step Fourier propagator.

Reproduces the Brewster-angle experiment of Sprouse & Awadallah, "An
angle-dependent impedance boundary condition for the split-step parabolic
equation method", IEEE TAP 60(2), 2012 (Figs. 2-4): a 2.25 GHz V-pol
Gaussian beam (1 deg half-width, -35 deg elevation, 25 m height) hits a
lossless eps = 2 ground near the Brewster angle. The split-step solutions
with the Leontovich condition (DMFT), the consistent-field condition with the
published splitting (eqs. 44-45) and with the even splitting weight are
compared with the exact plane-wave-spectrum solution.

    python scripts/ssf_impedance_cf_demo.py
"""
import math as fm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pywaveprop.helmholtz_common import HelmholtzMeshParams2D
from pywaveprop.helmholtz_jax import ConstWaveSpeedModel
from pywaveprop.helmholtz_ssf_jax import SSFHelmholtzPropagator, impedance_alpha
from tests.pywaveprop_jax.test_ssf_jax import _half_space_exact

freq = 2.25e9
k0 = 2 * fm.pi * freq / 3e8
eps, pol = 2.0, 'V'
h, bw, el = 25.0, fm.radians(1.0), fm.radians(-35.0)
dz = 3 * (2 * fm.pi / k0) / 8


def source(z):
    return fm.sqrt(k0) * fm.tan(bw) * np.exp(-0.5 * k0 ** 2 * (z - h) ** 2 * fm.tan(bw) ** 2
                                             + 1j * k0 * (z - h) * fm.sin(el))


def model(lower_bc, **kw):
    return SSFHelmholtzPropagator.create(
        freq_hz=freq, wave_speed=ConstWaveSpeedModel(3e8), kz_max=k0, k_bounds=(k0, k0), precision=0.01,
        mesh_params=HelmholtzMeshParams2D(x_size_m=150.0, z_size_m=100.0, dx_output_m=0.5, dz_output_m=dz),
        lower_bc=lower_bc, lower_impedance_alpha=impedance_alpha(k0, eps, pol), lower_eps=eps, polarization=pol,
        dx_m=0.5, dz_m=dz, **kw)


models = {'DMFT (Leontovich)': model('impedance'),
          'CF, paper split (eqs. 44-45)': model('impedance_cf', cf_split='paper'),
          'CF, even split': model('impedance_cf')}
m0 = next(iter(models.values()))
x, z = np.asarray(m0.x_output_grid()), np.asarray(m0.z_output_grid())
exact = _half_space_exact(k0, eps, pol, source, x, z)
init = _half_space_exact(k0, eps, pol, source, [0.0], np.asarray(m0.z_computational_grid()))[0]
fields = {'exact': exact, **{k: np.asarray(m.compute(init)) for k, m in models.items()}}

vmax = 20 * np.log10(np.abs(exact).max())
fig, axes = plt.subplots(2, 4, figsize=(22, 9))
for ax, (name, f) in zip(axes[0], fields.items()):
    ax.pcolormesh(x, z, 20 * np.log10(np.abs(f).T + 1e-30), shading='auto', cmap='jet', vmin=vmax - 80, vmax=vmax)
    err = '' if name == 'exact' else f'\nrel. L2 error {np.linalg.norm(f - exact) / np.linalg.norm(exact):.1e}'
    ax.set_title(name + err)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Height (m)')
for ax, (name, f) in zip(axes[1, 1:], list(fields.items())[1:]):
    ax.pcolormesh(x, z, 20 * np.log10(np.abs(f - exact).T + 1e-30), shading='auto', cmap='jet',
                  vmin=vmax - 80, vmax=vmax)
    ax.set_title(f'|error|: {name}')
    ax.set_xlabel('Range (m)')
ix = int(np.argmin(np.abs(x - 140.0)))
for name, f in fields.items():
    axes[1, 0].plot(z, 20 * np.log10(np.abs(f[ix]) + 1e-30), '--k' if name == 'exact' else '-', label=name)
axes[1, 0].set_xlim(50, 100)
axes[1, 0].set_ylim(vmax - 70, vmax - 30)
axes[1, 0].set_title('Reflected beam at x = 140 m (Brewster null)')
axes[1, 0].set_xlabel('Height (m)')
axes[1, 0].set_ylabel('|u| (dB)')
axes[1, 0].legend()
plt.tight_layout()
plt.savefig('ssf_impedance_cf_brewster.png', dpi=110)
for name, f in list(fields.items())[1:]:
    print(f'{name}: relative L2 error {np.linalg.norm(f - exact) / np.linalg.norm(exact):.2e}')
