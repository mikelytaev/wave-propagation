import numpy as np

import pyximport
import cmath as cm
import math as fm
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
import propagators._utils as utils
import propagators.dispersion_relations as disp_rels
from scipy.optimize import differential_evolution, NonlinearConstraint
import matplotlib.pyplot as plt
import mpmath
import opt_utils


def opt_coefs_to_coefs(coefs_arr, order):
    n = order[0]
    m = order[1]
    num_coefs = np.array([coefs_arr[2 + 2 * i] + 1j * coefs_arr[2 + 2 * i + 1] for i in range(0, n)])
    den_coefs = np.array([coefs_arr[2+ 2 * n + 2 * i] + 1j * coefs_arr[2 + 2 * n + 2 * i + 1] for i in range(0, m)])
    return num_coefs, den_coefs



def opt_coefs_to_grids(coefs_arr):
    dx = coefs_arr[0]
    dz = coefs_arr[1]
    return dx, dz


def coefs_to_opt_coefs(coefs):
    co = []
    for c in coefs:
        co += [c[0].real, c[0].imag]
    for c in coefs:
        co += [c[1].real, c[1].imag]
    return co


k0 = 2*cm.pi
theta_max_degrees = 85
order = (7, 8)

print(k0*fm.sin(theta_max_degrees * fm.pi / 180))


def fit_func(coefs_arr):
    dx, dz = opt_coefs_to_grids(coefs_arr)
    return 1 / (dx * dz)


eps = 1e-3
eps_x_max = 1e3


def constraint_ga(coefs_arr):
    dx, dz = opt_coefs_to_grids(coefs_arr)
    num_coefs, den_coefs = opt_coefs_to_coefs(coefs_arr, order)
    err = disp_rels.k_x_abs_error_range(2 * cm.pi, dx, dz, num_coefs, den_coefs, k0 * fm.sin(theta_max_degrees * fm.pi / 180),
                                        round(theta_max_degrees) * 5) / dx
    return err


def constraint_ga2(coefs_arr):
    dx, dz = opt_coefs_to_grids(coefs_arr)
    num_coefs, den_coefs = opt_coefs_to_coefs(coefs_arr, order)
    min_im = disp_rels.k_x_min_im(2 * cm.pi, dx, dz, num_coefs, den_coefs, k0 * fm.sin(theta_max_degrees * fm.pi / 180),
                                        round(theta_max_degrees) * 5)
    return min_im


def constraint_pade_2nd_order(coefs_arr):
    dx, dz = opt_coefs_to_grids(coefs_arr)
    pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2*cm.pi, dx=dx)
    num_coefs = np.array([a[0] for a in pade_coefs])
    den_coefs = np.array([a[1] for a in pade_coefs])
    err = disp_rels.k_x_abs_error_range(2 * cm.pi, dx, dz, num_coefs, den_coefs, k0 * fm.sin(theta_max_degrees * fm.pi / 180),
                                        round(theta_max_degrees) * 5) / dx
    return err


def constraint_pade_joined_order(coefs_arr):
    dx, dz = opt_coefs_to_grids(coefs_arr)

    def diff2(s):
        return mpmath.acosh(1 + (k0 * dz) ** 2 * s / 2) ** 2 / (k0 * dz) ** 2
    pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=diff2, k0=2*cm.pi, dx=dx)
    num_coefs = np.array([a[0] for a in pade_coefs])
    den_coefs = np.array([a[1] for a in pade_coefs])
    err = disp_rels.k_x_abs_error_range(2 * cm.pi, dx, dz, num_coefs, den_coefs, k0 * fm.sin(theta_max_degrees * fm.pi / 180),
                                        round(theta_max_degrees) * 5) / dx
    return err


bounds_pade = [(0.1, 0.1), (0.00001, 0.0001)]

# result_joined_pade = differential_evolution(
#     fit_func,
#     bounds_pade,
#     constraints=(NonlinearConstraint(constraint_pade_joined_order, 0, eps/eps_x_max)),
#     popsize=15,
#     disp=True,
#     recombination=0.7,
#     strategy='best1bin',
#     tol=1e-9,
#     polish=False,
#     maxiter=2000)
# print(result_joined_pade)
# dx_joined_pade, dz_joined_pade = opt_coefs_to_grids(result_joined_pade.x)
dx_joined_pade, dz_joined_pade = 0.5, 0.5
dx_ga, dz_ga = dx_joined_pade, dz_joined_pade * 1

bounds_ga = [(-1000, 1000)] * (order[0] + order[1]) * 2

result_ga = differential_evolution(
    func=opt_utils.fit_func_ga,
    args=(dx_ga, dz_ga, order, theta_max_degrees),
    bounds=bounds_ga,
    popsize=100,
    disp=True,
    recombination=1,
    strategy='randtobest1exp',
    tol=1e-9,
    maxiter=50000,
    polish=False,
    workers=4
)
print(result_ga)

num_coefs_ga, den_coefs_ga = opt_coefs_to_coefs_ga(result_ga.x, order)

result_pade = differential_evolution(fit_func, bounds_pade, constraints=(NonlinearConstraint(constraint_pade_2nd_order, 0, eps/eps_x_max)), popsize=15, disp=True, recombination=0.99, strategy='currenttobest1bin', tol=1e-9, maxiter=2000)
print(result_pade)

dx_pade, dz_pade = opt_coefs_to_grids(result_pade.x)



pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2*cm.pi, dx=dx_pade)


def diff2(s):
    return mpmath.acosh(1 + (k0 * dz_joined_pade) ** 2 * s / 2) ** 2 / (k0 * dz_joined_pade) ** 2


joined_pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=diff2, k0=2*cm.pi, dx=dx_joined_pade)


def k_x_angle(dx, dz, num_coefs, den_coefs, kz_arr):
    return np.array([disp_rels.discrete_k_x(k0, dx, dz, num_coefs, den_coefs, kz) for kz in kz_arr])


kz_arr = np.linspace(0, 2*k0, 10000)
k_x_r = np.array([cm.sqrt(k0**2 - kz**2) for kz in kz_arr])
k_x_1 = k_x_angle(dx_ga, dz_ga, num_coefs_ga, den_coefs_ga, kz_arr)
pade_coefs_num = np.array([a[0] for a in pade_coefs])
pade_coefs_den = np.array([a[1] for a in pade_coefs])
joined_pade_coefs_num = np.array([a[0] for a in joined_pade_coefs])
joined_pade_coefs_den = np.array([a[1] for a in joined_pade_coefs])
k_x_2 = k_x_angle(dx_pade, dz_pade, pade_coefs_num, pade_coefs_den, kz_arr)
k_x_3 = k_x_angle(dx_joined_pade, dz_joined_pade, joined_pade_coefs_num, joined_pade_coefs_den, kz_arr)

plt.figure(figsize=(6, 3.2))
plt.plot(kz_arr, (np.abs(k_x_1 - k_x_r)), label='opt')
plt.plot(kz_arr, (np.abs(k_x_2 - k_x_r)), label='Pade')
plt.plot(kz_arr, (np.abs(k_x_3 - k_x_r)), label='Joined Pade')
plt.xlabel('k_z')
plt.ylabel('k_x abs. error')
plt.xlim([kz_arr[0], kz_arr[-1]])
#plt.ylim([1e-10, 1e-1])
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



plt.figure(figsize=(6, 3.2))
# plt.plot(angles, (np.real(k_x_1)), label='opt real')
# plt.plot(angles, (np.real(k_x_2)), label='Pade real')
plt.plot(kz_arr, (np.imag(k_x_1)), label='opt imag')
plt.plot(kz_arr, (np.imag(k_x_2)), label='Pade imag')
plt.plot(kz_arr, (np.imag(k_x_3)), label='Joined Pade imag')
plt.xlabel('k_z')
plt.ylabel('k_x abs. error')
plt.xlim([kz_arr[0], kz_arr[-1]])
#plt.ylim([1e-10, 1e-1])
#plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



from rwp.sspade import *
from rwp.vis import *
import matplotlib as mpl


logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 3000
env.terrain = Terrain(ground_material=PerfectlyElectricConducting())
#env.knife_edges = [KnifeEdge(range=1.5e3, height=100)]

ant = GaussAntenna(freq_hz=300e6, height=1500, beam_width=30, eval_angle=0, polarz='H')

max_propagation_angle = theta_max_degrees
max_range_m = eps_x_max


pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=2,
                                                      exp_pade_order=order,
                                                      dx_wl=dx_pade,
                                                      x_output_filter=4,
                                                      dz_wl=dz_pade,
                                                      z_output_filter=8,
                                                      two_way=False
                                                  ))
pade_field = pade_task.calculate()

joined_pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=5,
                                                      exp_pade_order=order,
                                                      dx_wl=dx_joined_pade,
                                                      x_output_filter=4,
                                                      dz_wl=dz_joined_pade,
                                                      z_output_filter=8,
                                                      two_way=False
                                                  ))
joined_pade_field = joined_pade_task.calculate()

opt_pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=2,
                                                      exp_pade_coefs=list(zip_longest(num_coefs_ga, den_coefs_ga, fillvalue=0.0j)),
                                                      dx_wl=dx_ga,
                                                      x_output_filter=4,
                                                      dz_wl=dz_ga,
                                                      z_output_filter=8,
                                                      two_way=False,
                                                      inv_z_transform_rtol=1e-7
                                                  ))
opt_pade_field = opt_pade_task.calculate()

pade_vis = FieldVisualiser(pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3, label="Pade")
plt = pade_vis.plot2d(min=-70, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

joined_pade_vis = FieldVisualiser(joined_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3, label="Joined Pade")
plt = joined_pade_vis.plot2d(min=-70, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

opt_vis = FieldVisualiser(opt_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3, label="Opt")
plt = opt_vis.plot2d(min=-70, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

opt_vis.plot_hor(300, pade_vis, joined_pade_vis)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

f, ax1 = plt.subplots(1, 1, sharey=True)
opt_vis.plot_ver(max_range_m, ax1, pade_vis, joined_pade_vis)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()
