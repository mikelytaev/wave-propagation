import numpy as np

import pyximport
import cmath as cm
import math as fm
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
import propagators._utils as utils
import propagators.dispersion_relations as disp_rels
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


def opt_coefs_to_coefs(coefs_arr, order):
    n = order[0]
    m = order[1]
    num_coefs = np.array([coefs_arr[2 * i] + 1j * coefs_arr[2 * i + 1] for i in range(0, n)])
    den_coefs = np.array([coefs_arr[2 * n + 2 * i] + 1j * coefs_arr[2 * n + 2 * i + 1] for i in range(0, m)])
    return num_coefs, den_coefs


def coefs_to_opt_coefs(coefs):
    co = []
    for c in coefs:
        co += [c[0].real, c[0].imag]
    for c in coefs:
        co += [c[1].real, c[1].imag]
    return co


dx = 1
theta_max_degrees = 85
order = (6, 7)


def fit_func(coefs_arr):
    num_coefs, den_coefs = opt_coefs_to_coefs(coefs_arr, order)
    err = disp_rels.k_x_abs_error_range(2*cm.pi, dx, num_coefs, den_coefs, theta_max_degrees, round(theta_max_degrees)*2)
    return err


pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2*cm.pi, dx=dx)
x0 = coefs_to_opt_coefs(pade_coefs)

bounds = [(-5, 5)] * (order[0] + order[1]) * 2

result = differential_evolution(fit_func, bounds, popsize=15, disp=True, recombination=0.9, strategy='currenttobest1bin', tol=1e-9, maxiter=50000)
print(result)

num_coefs, den_coefs = opt_coefs_to_coefs(result.x, order)


def k_x_angle(dx, num_coefs, den_coefs, thetas):
    return np.array([disp_rels.discrete_k_x(k0, dx, num_coefs, den_coefs, theta) for theta in thetas])


k0 = 2*cm.pi
angles = np.linspace(0, 90, 1000)
k_x_r = np.array([cm.sqrt(k0**2 - kz**2) for kz in k0*np.sin(angles*fm.pi/180)])
k_x_1 = k_x_angle(dx, num_coefs, den_coefs, angles)
pade_coefs_num = np.array([a[0] for a in pade_coefs])
pade_coefs_den = np.array([a[1] for a in pade_coefs])
k_x_2 = k_x_angle(dx, pade_coefs_num, pade_coefs_den, angles)

plt.figure(figsize=(6, 3.2))
plt.plot(angles, (np.abs(k_x_1 - k_x_r)), label='opt')
plt.plot(angles, (np.abs(k_x_2 - k_x_r)), label='Pade')
plt.xlabel('Propagation angle, degrees')
plt.ylabel('k_x abs. error')
plt.xlim([0, 90])
#plt.ylim([1e-10, 1e-1])
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from rwp.sspade import *
from rwp.vis import *
import matplotlib as mpl


logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 300
env.terrain = Terrain(ground_material=PerfectlyElectricConducting())
env.knife_edges = [KnifeEdge(range=1.5e3, height=100)]

ant = GaussAntenna(freq_hz=300e6, height=100, beam_width=3, eval_angle=0, polarz='H')

max_propagation_angle = 10
max_range_m = 3.0e3


pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=4,
                                                      exp_pade_order=order,
                                                      dx_wl=dx,
                                                      x_output_filter=4,
                                                      dz_wl=0.1,
                                                      z_output_filter=8,
                                                      two_way=False
                                                  ))
pade_field = pade_task.calculate()

opt_pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=4,
                                                      exp_pade_coefs=list(zip_longest(num_coefs, den_coefs, fillvalue=0.0j)),
                                                      dx_wl=dx,
                                                      x_output_filter=4,
                                                      dz_wl=0.1,
                                                      z_output_filter=8,
                                                      two_way=False
                                                  ))
opt_pade_field = opt_pade_task.calculate()

pade_vis = FieldVisualiser(pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3)
plt = pade_vis.plot2d(min=-120, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

opt_vis = FieldVisualiser(opt_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3)
plt = opt_vis.plot2d(min=-120, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()