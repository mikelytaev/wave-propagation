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


k0 = 2*cm.pi
theta_max_degrees = 85
order = (7, 8)

print(k0*fm.sin(theta_max_degrees * fm.pi / 180))

eps = 1e-3
eps_x_max = 1e3


dx_joined_pade, dz_joined_pade = 2, 0.5
dx_ga, dz_ga = dx_joined_pade, dz_joined_pade * 1

bounds_ga = [(-10, 10)] * (order[0] + order[1]) * 2

result_ga = differential_evolution(
    func=opt_utils.fit_func_ga,
    args=(dx_ga, dz_ga, order, theta_max_degrees),
    bounds=bounds_ga,
    popsize=50,
    disp=True,
    recombination=1,
    strategy='randtobest1exp',
    tol=1e-9,
    maxiter=30000,
    polish=False,
    workers=-1,
    #callback=lambda xk, convergence: print(xk)
)
print(result_ga)

num_coefs_ga, den_coefs_ga = opt_utils.opt_coefs_to_coefs_ga(result_ga.x, order)

def diff2(s):
    return mpmath.acosh(1 + (k0 * dz_joined_pade) ** 2 * s / 2) ** 2 / (k0 * dz_joined_pade) ** 2


joined_pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=diff2, k0=2*cm.pi, dx=dx_joined_pade)


def k_x_angle(dx, dz, num_coefs, den_coefs, kz_arr):
    return np.array([disp_rels.discrete_k_x(k0, dx, dz, num_coefs, den_coefs, kz) for kz in kz_arr])


kz_arr = np.linspace(0, 2*k0, 10000)
k_x_r = np.array([cm.sqrt(k0**2 - kz**2) for kz in kz_arr])
k_x_1 = k_x_angle(dx_ga, dz_ga, num_coefs_ga, den_coefs_ga, kz_arr)
joined_pade_coefs_num = np.array([a[0] for a in joined_pade_coefs])
joined_pade_coefs_den = np.array([a[1] for a in joined_pade_coefs])
k_x_3 = k_x_angle(dx_joined_pade, dz_joined_pade, joined_pade_coefs_num, joined_pade_coefs_den, kz_arr)

plt.figure(figsize=(6, 3.2))
plt.plot(kz_arr, (np.abs(k_x_1 - k_x_r)), label='opt')
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
env.z_max = 500
env.terrain = Terrain(ground_material=PerfectlyElectricConducting())
env.knife_edges = [KnifeEdge(range=0.5e3, height=env.z_max/2)]

ant = GaussAntenna(freq_hz=300e6, height=env.z_max/2, beam_width=theta_max_degrees/2, eval_angle=0, polarz='H')

max_propagation_angle = theta_max_degrees
max_range_m = eps_x_max

joined_pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=5,
                                                      exp_pade_order=order,
                                                      dx_wl=dx_joined_pade,
                                                      x_output_filter=1,
                                                      dz_wl=dz_joined_pade,
                                                      z_output_filter=1,
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
                                                      x_output_filter=1,
                                                      dz_wl=dz_ga,
                                                      z_output_filter=1,
                                                      two_way=False,
                                                      inv_z_transform_rtol=1e-4
                                                  ))
opt_pade_field = opt_pade_task.calculate()

joined_pade_vis = FieldVisualiser(joined_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3, label="Joined Pade")
plt = joined_pade_vis.plot2d(min=-160, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

opt_vis = FieldVisualiser(opt_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3, label="Opt")
plt = opt_vis.plot2d(min=-160, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

opt_vis.plot_hor(250, joined_pade_vis)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

f, ax1 = plt.subplots(1, 1, sharey=True)
opt_vis.plot_ver(max_range_m, ax1, joined_pade_vis)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()
