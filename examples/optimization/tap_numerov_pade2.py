import numpy as np
import cmath as cm
import math as fm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def elev_int_1d(x, angle, height, r):
    length = height / fm.tan(angle * cm.pi / 180)
    if r <= x <= r + length:
        return (x - r) * fm.tan(angle * cm.pi / 180)
    elif r + length < x <= r + 2*length:
        return (r + 2*length - x) * fm.tan(angle * cm.pi / 180)
    else:
        return 0

# x_grid = np.linspace(0, 20000, 1000)
# plt.plot(x_grid, [elev_int_1d(v) for v in x_grid])
# plt.show()


from rwp.sspade import *
from rwp.vis import *
#from rwp.petool import PETOOLPropagationTask

logging.basicConfig(level=logging.DEBUG)
env = Troposphere()
env.z_max = 300
env.terrain = Terrain(elevation=lambda x: elev_int_1d(x, 5, 200, 7e3) + elev_int_1d(x, 5, 200, 14e3), ground_material=PerfectlyElectricConducting())
#env.knife_edges = [KnifeEdge(80e3, 200)]
profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 40], fill_value="extrapolate")
#env.M_profile = lambda x, z: profile1d(z)

ant = GaussAntenna(freq_hz=3000e6, height=70, beam_width=4, eval_angle=0, polarz='H')
max_range_m = 20000
pade_task_4 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=5,
                                                      #modify_grid=True,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=4,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      dz_wl=1.4,
                                                      #two_way=False,
                                                      storage=PickleStorage()
                                                  ))
pade_field_4 = pade_task_4.calculate()

# pade_task_4f = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
#                                                   HelmholtzPropagatorComputationalParams(
#                                                       terrain_method=TerrainMethod.staircase,
#                                                       max_propagation_angle=5,
#                                                       #modify_grid=True,
#                                                       grid_optimizator_abs_threshold=5e-3,
#                                                       z_order=4,
#                                                       exp_pade_order=(10, 11),
#                                                       dx_wl=500,
#                                                       dz_wl=0.8*4.5,
#                                                       #two_way=False,
#                                                       storage=PickleStorage()
#                                                   ))
# pade_field_4f = pade_task_4f.calculate()

# pade_task_2 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=150e3, comp_params=
#                                                   HelmholtzPropagatorComputationalParams(
#                                                       terrain_method=TerrainMethod.staircase,
#                                                       max_propagation_angle=5,
#                                                       modify_grid=False,
#                                                       grid_optimizator_abs_threshold=5e-3,
#                                                       z_order=2,
#                                                       exp_pade_order=(10, 11),
#                                                       dx_wl=500,
#                                                       dz_wl=0.1,
#                                                       storage=PickleStorage()
#                                                   ))
# pade_field_2 = pade_task_2.calculate()

pade_task_2f = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=5,
                                                      #modify_grid=True,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      dz_wl=1.4,
                                                      #two_way=False,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2f = pade_task_2f.calculate()

#env.terrain = Terrain(lambda x: elev_int_1d(x)-0.00001)
# petool_task = PETOOLPropagationTask(antenna=ant, env=env, two_way=False, max_range_m=150e3, dx_wl=500, n_dx_out=1, dz_wl=1)
# petool_field = petool_task.calculate()

# env.z_max = 3000
# petool_task_m = PETOOLPropagationTask(antenna=ant, env=env, two_way=False, max_range_m=150e3, dx_wl=500, n_dx_out=1,
#                                       dz_wl=3, n_dz_out=2)
# petool_field_m = petool_task_m.calculate()

pade_vis_4 = FieldVisualiser(pade_field_4, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=500, dz=0.8 (4th order)', x_mult=1E-3)
# pade_vis_4f = FieldVisualiser(pade_field_4f, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
#                              label='Pade-[10/11], dx=500, dz=3.6 (4th order)', x_mult=1E-3)
# pade_vis_2 = FieldVisualiser(pade_field_2, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
#                              label='Pade-[10/11], dx=500, dz=0.1 (2th order)', x_mult=1E-3)
pade_vis_2f = FieldVisualiser(pade_field_2f, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=500, dz=3.6 (2th order)', x_mult=1E-3)
# petool_vis = FieldVisualiser(petool_field, env=env, trans_func=lambda x: x, label='SSF (PETOOL)', x_mult=1E-3)
# petool_vis_m = FieldVisualiser(petool_field_m, env=env, trans_func=lambda x: x, label='SSF (PETOOL) z_max=3000 m', x_mult=1E-3)

plt = pade_vis_4.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

# plt = pade_vis_4f.plot2d(min=-100, max=0, show_terrain=True)
# plt.xlabel('Range (km)')
# plt.ylabel('Height (m)')
# plt.tight_layout()
# plt.show()

# plt = pade_vis_2.plot2d(min=-100, max=0, show_terrain=True)
# plt.xlabel('Range (km)')
# plt.ylabel('Height (m)')
# plt.tight_layout()
# plt.show()

plt = pade_vis_2f.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

# plt = petool_vis.plot2d(min=-100, max=0, show_terrain=True)
# plt.xlabel('Range (km)')
# plt.ylabel('Height (m)')
# plt.tight_layout()
# plt.show()
#
# plt = petool_vis_m.plot2d(min=-100, max=0, show_terrain=True)
# plt.xlabel('Range (km)')
# plt.ylabel('Height (m)')
# plt.ylim([0, 300])
# plt.tight_layout()
# plt.show()

plt = pade_vis_4.plot_hor_over_terrain(1, pade_vis_2f)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.xlim([0.5, 20])
plt.ylim([-100, 0])
plt.grid(True)
plt.tight_layout()
plt.show()

f, (ax1) = plt.subplots(1, 1, sharey=True)
pade_vis_4.plot_ver(5 * 1E3, ax1, pade_vis_2f)
ax1.set_ylabel('Height (m)')
ax1.set_xlabel('10log|u| (dB)')
ax1.grid()
f.tight_layout()
f.show()

f, (ax1) = plt.subplots(1, 1, sharey=True)
pade_vis_4.plot_ver(150 * 1E3, ax1, pade_vis_2f)
ax1.set_ylabel('Height (m)')
ax1.set_xlabel('10log|u| (dB)')
ax1.grid()
f.tight_layout()
f.show()
