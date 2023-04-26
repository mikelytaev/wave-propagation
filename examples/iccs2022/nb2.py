import numpy as np

import pyximport
import cmath as cm
import math as fm
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
import propagators._utils as utils
import propagators.dispersion_relations as disp_rels
from scipy.optimize import differential_evolution, NonlinearConstraint
import matplotlib.pyplot as plt


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
theta_max_degrees = 22
order = (6, 7)

print(k0*fm.sin(theta_max_degrees * fm.pi / 180))


def fit_func(coefs_arr):
    dx, dz = opt_coefs_to_grids(coefs_arr)
    return 1 / (dx * dz)


eps = 3e-4
eps_x_max = 3e3


def constraint_ga(coefs_arr):
    dx, dz = opt_coefs_to_grids(coefs_arr)
    num_coefs, den_coefs = opt_coefs_to_coefs(coefs_arr, order)
    err = disp_rels.k_x_abs_error_range(2 * cm.pi, dx, dz, num_coefs, den_coefs, k0 * fm.sin(theta_max_degrees * fm.pi / 180),
                                        round(theta_max_degrees) * 5) / dx / k0
    return err


def constraint_pade_2nd_order(coefs_arr):
    dx, dz = opt_coefs_to_grids(coefs_arr)
    pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2*cm.pi, dx=dx)
    num_coefs = np.array([a[0] for a in pade_coefs])
    den_coefs = np.array([a[1] for a in pade_coefs])
    err = disp_rels.k_x_abs_error_range(2 * cm.pi, dx, dz, num_coefs, den_coefs, k0 * fm.sin(theta_max_degrees * fm.pi / 180),
                                        round(theta_max_degrees) * 5) / dx / k0
    return err


bounds_pade = [(0.1, 100), (0.0001, 1)]

result_pade = differential_evolution(
    fit_func,
    bounds_pade,
    constraints=(NonlinearConstraint(constraint_pade_2nd_order, 0, eps/eps_x_max)),
    popsize=15,
    disp=True,
    recombination=1,
    strategy='randtobest1exp',
    tol=1e-9,
    polish=False,
    maxiter=2000,
    workers=1,
    callback=lambda xk, convergence: print(xk)
)
print(result_pade)
dx_pade, dz_pade = opt_coefs_to_grids(result_pade.x)
pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2*cm.pi, dx=dx_pade)
pade_coefs_num = np.array([a[0] for a in pade_coefs])
pade_coefs_den = np.array([a[1] for a in pade_coefs])

bounds_ga = bounds_pade + [(-100, 100)] * (order[0] + order[1]) * 2

result_ga = differential_evolution(
    fit_func,
    bounds_ga,
    constraints=(NonlinearConstraint(constraint_ga, 0, eps/eps_x_max)),
    popsize=15,
    disp=True,
    recombination=1.0,
    strategy='randtobest1exp',
    tol=1e-9,
    maxiter=4000,
    polish=False,
    workers=1,
    callback=lambda xk, convergence: print(str(constraint_ga(xk)) + " " + str(opt_coefs_to_grids(xk)))
)
print(result_ga)
dx_ga, dz_ga = opt_coefs_to_grids(result_ga.x)
ga_coefs_num, ga_coefs_den = opt_coefs_to_coefs(result_ga.x, order)


def k_x_angle(dx, dz, num_coefs, den_coefs, kz_arr):
    return np.array([disp_rels.discrete_k_x(k0, dx, dz, num_coefs, den_coefs, kz) for kz in kz_arr])


angles = np.linspace(0, theta_max_degrees*1.5, 1000)
kz_arr = k0*np.sin(angles*fm.pi/180)
k_x_r = np.sqrt(k0**2 - kz_arr**2)
k_x_ga = k_x_angle(dx_ga, dz_ga, ga_coefs_num, ga_coefs_den, kz_arr)
k_x_pade = k_x_angle(dx_pade, dz_pade, pade_coefs_num, pade_coefs_den, kz_arr)

k_x_ga_error = np.abs(k_x_ga - k_x_r) / k0 / dx_ga * eps_x_max
k_x_pade_error = np.abs(k_x_pade - k_x_r) / k0 / dx_pade * eps_x_max

plt.figure(figsize=(6, 3.2))
plt.plot(angles, k_x_ga_error, label='Diff. evol.')
plt.plot(angles, k_x_pade_error, label='Pade')
plt.xlabel('Angle (degrees)')
plt.ylabel('k_x abs. error')
plt.xlim([angles[0], angles[-1]])
plt.ylim([1e-6, 1e0])
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.axvline(x=theta_max_degrees, color='gray', linestyle='--')
plt.tight_layout()
plt.savefig("ex2_k_x_error.eps")
plt.show()


kz_arr = k0*np.linspace(0, 1.5, 1000)
k_x_r = np.sqrt(k0**2 - kz_arr**2)
k_x_ga = k_x_angle(dx_ga, dz_ga, ga_coefs_num, ga_coefs_den, kz_arr)
k_x_pade = k_x_angle(dx_pade, dz_pade, pade_coefs_num, pade_coefs_den, kz_arr)
plt.figure(figsize=(6, 3.2))
plt.plot(kz_arr/k0, np.imag(k_x_ga)/k0, label='Diff. evol.')
plt.plot(kz_arr/k0, np.imag(k_x_pade)/k0, label='Pade')
plt.xlabel('k_z/k')
plt.ylabel('Im(k_x)/k')
plt.xlim([kz_arr[0]/k0, kz_arr[-1]/k0])
#plt.ylim([1e-10, 1e-1])
#plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ex2_k_x.eps")
plt.show()



from rwp.sspade import *
from rwp.vis import *
import matplotlib as mpl


logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 300
elevation = lambda x: pyramid2(x, 20, 200, 1.5e3)
env.terrain = Terrain(elevation=elevation, ground_material=PerfectlyElectricConducting())
#env.knife_edges = [KnifeEdge(range=1.5e3, height=1500)]

ant = GaussAntenna(freq_hz=1000e6, height=200, beam_width=theta_max_degrees-5, eval_angle=0, polarz='H')

max_propagation_angle = theta_max_degrees
max_range_m = eps_x_max


pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=2,
                                                      exp_pade_order=order,
                                                      dx_wl=dx_ga / round(dx_ga / dx_pade),
                                                      x_output_filter=round(dx_ga / dx_pade),
                                                      dz_wl=dz_ga / round(dz_ga / dz_pade),
                                                      z_output_filter=round(dz_ga / dz_pade),
                                                      two_way=False
                                                  ))
pade_field = pade_task.calculate()

pade_task_f = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=2,
                                                      exp_pade_order=order,
                                                      dx_wl=dx_ga,
                                                      x_output_filter=1,
                                                      dz_wl=dz_ga,
                                                      z_output_filter=1,
                                                      two_way=False
                                                  ))
pade_field_f = pade_task_f.calculate()

opt_pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=2,
                                                      exp_pade_coefs=list(zip_longest(ga_coefs_num, ga_coefs_den, fillvalue=0.0j)),
                                                      dx_wl=dx_ga,
                                                      x_output_filter=1,
                                                      dz_wl=dz_ga,
                                                      z_output_filter=1,
                                                      two_way=False,
                                                      inv_z_transform_rtol=1e-7
                                                  ))
opt_pade_field = opt_pade_task.calculate()

pade_vis = FieldVisualiser(pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3, label="Pade")
plt = pade_vis.plot2d(min=-120, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.savefig("ex3_pade.eps")
plt.show()

pade_f_vis = FieldVisualiser(pade_field_f, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3, label="Pade")
plt = pade_f_vis.plot2d(min=-120, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.savefig("ex3_pade_f.eps")
plt.show()

opt_vis = FieldVisualiser(opt_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3, label="Opt")
plt = opt_vis.plot2d(min=-120, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.savefig("ex3_ga.eps")
plt.show()

opt_vis.plot_hor(300, pade_vis)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

f, ax1 = plt.subplots(1, 1, sharey=True)
opt_vis.plot_ver(max_range_m, ax1, pade_vis)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()
