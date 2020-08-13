import numpy as np
import cmath as cm
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pyproj


bounds = (29.636637, 60.112502, 31.288706, 60.699130)
west, south, east, north = bounds

import pickle

with open('elevation.pickle', 'rb') as f:
    elevation = pickle.load(f)

geod = pyproj.Geod(ellps='WGS84')

azimuth1, azimuth2, distance = geod.inv(west, south, east, north)

long_grid = np.linspace(west, east, elevation.shape[0])
lat_grid = np.linspace(north, south, elevation.shape[1])

elev_int = interp2d(long_grid, lat_grid, elevation.T)

def elev_int_1d(x):
    if not 0 < x < distance:
        return 0.0
    v = x / distance
    return max(elev_int(west + (east - west) * v, south + (north - south) * v)[0], 0)


from rwp.sspade import *
from rwp.vis import *
#from rwp.petool import PETOOLPropagationTask

logging.basicConfig(level=logging.DEBUG)
env = Troposphere()
env.z_max = 3000
env.terrain = Terrain(elevation=elev_int_1d, ground_material=FreshWater())
#env.knife_edges = [KnifeEdge(80e3, 200)]
profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 40], fill_value="extrapolate")
#env.M_profile = lambda x, z: profile1d(z)

ant = GaussAntenna(freq_hz=3000e6, height=70, beam_width=4, eval_angle=0, polarz='H')

pade_task_4 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=150e3, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=5,
                                                      #modify_grid=True,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=4,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      dz_wl=0.8,
                                                      #two_way=False,
                                                      storage=PickleStorage()
                                                  ))
pade_field_4 = pade_task_4.calculate()

pade_task_4f = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=150e3, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=5,
                                                      modify_grid=True,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=4,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      dz_wl=3,
                                                      #two_way=False,
                                                      storage=PickleStorage()
                                                  ))
pade_field_4f = pade_task_4f.calculate()

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

pade_task_2f = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=150e3, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=5,
                                                      modify_grid=True,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      dz_wl=3,
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
pade_vis_4f = FieldVisualiser(pade_field_4f, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=500, dz=4 (4th order)', x_mult=1E-3)
# pade_vis_2 = FieldVisualiser(pade_field_2, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
#                              label='Pade-[10/11], dx=500, dz=0.1 (2th order)', x_mult=1E-3)
pade_vis_2f = FieldVisualiser(pade_field_2f, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=500, dz=4 (2th order)', x_mult=1E-3)
# petool_vis = FieldVisualiser(petool_field, env=env, trans_func=lambda x: x, label='SSF (PETOOL)', x_mult=1E-3)
# petool_vis_m = FieldVisualiser(petool_field_m, env=env, trans_func=lambda x: x, label='SSF (PETOOL) z_max=3000 m', x_mult=1E-3)

plt = pade_vis_4.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = pade_vis_4f.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

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

plt = pade_vis_4.plot_hor_over_terrain(10, pade_vis_2f, pade_vis_4f)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.xlim([0.5, 150])
plt.ylim([-100, 0])
plt.grid(True)
plt.tight_layout()
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
pade_vis_4.plot_ver(10 * 1E3, ax1, pade_vis_2f, pade_vis_4f)
ax1.set_ylabel('Height (m)')
ax1.set_xlabel('10log|u| (dB)')

pade_vis_4.plot_ver(1 * 1E3, ax2, pade_vis_2f, pade_vis_4f)
ax2.set_ylabel('Height (m)')
ax2.set_xlabel('10log|u| (dB)')
f.tight_layout()
f.show()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot([a / 6371000 * 1E6 for a in pade_field_4.z_grid], pade_field_4.z_grid)
#ax1.legend()
ax1.set_xlabel('M-units')
ax1.set_ylabel('Height (m)')

ax2.plot([profile1d(a) for a in pade_field_4.z_grid], pade_field_4.z_grid)
#ax2.legend()
ax2.set_xlabel('M-units')
ax2.set_ylabel('Height (m)')
f.tight_layout()
ax1.grid()
ax2.grid()
f.show()