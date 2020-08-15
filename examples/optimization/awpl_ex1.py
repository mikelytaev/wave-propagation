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
from rwp.petool import PETOOLPropagationTask

logging.basicConfig(level=logging.DEBUG)
env = Troposphere()
env.z_max = 300
env.terrain = Terrain(elevation=elev_int_1d, ground_material=FreshWater())
profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 40], fill_value="extrapolate")

ant = GaussAntenna(freq_hz=3000e6, height=70, beam_width=4, eval_angle=0, polarz='H')

pade_task_4 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=150e3, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=5,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=4,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      dz_wl=0.8,
                                                      storage=PickleStorage()
                                                  ))
pade_field_4 = pade_task_4.calculate()

petool_task = PETOOLPropagationTask(antenna=ant, env=env, two_way=False, max_range_m=150e3, dx_wl=500, n_dx_out=1, dz_wl=3)
petool_field = petool_task.calculate()

env.z_max = 3000
petool_task_m = PETOOLPropagationTask(antenna=ant, env=env, two_way=False, max_range_m=150e3, dx_wl=500, n_dx_out=1,
                                      dz_wl=3, n_dz_out=2)
petool_field_m = petool_task_m.calculate()

pade_vis_4 = FieldVisualiser(pade_field_4, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=500, dz=0.8 (4th order)', x_mult=1E-3)
petool_vis = FieldVisualiser(petool_field, env=env, trans_func=lambda x: x, label='SSF (PETOOL)', x_mult=1E-3)
petool_vis_m = FieldVisualiser(petool_field_m, env=env, trans_func=lambda x: x, label='SSF (PETOOL) z_max=3000 m', x_mult=1E-3)

plt = pade_vis_4.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = petool_vis.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = pade_vis_4.plot_hor_over_terrain(10, petool_vis, petool_vis_m)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.xlim([0.5, 150])
plt.ylim([-100, 0])
plt.grid(True)
plt.tight_layout()
plt.show()

norm = Normalize(-100, 0)
extent = [pade_vis_4.x_grid[0], pade_vis_4.x_grid[-1], pade_vis_4.z_grid[0], pade_vis_4.z_grid[-1]]
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 3.2))
ax1.imshow(pade_vis_4.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
im = ax2.imshow(petool_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
f.colorbar(im, fraction=0.046, pad=0.04)
terrain_grid = np.array([pade_vis_4.env.terrain.elevation(v) for v in pade_vis_4.x_grid / pade_vis_4.x_mult])
ax1.plot(pade_vis_4.x_grid, terrain_grid, 'k')
ax1.fill_between(pade_vis_4.x_grid, terrain_grid*0, terrain_grid, color='black')
ax2.plot(pade_vis_4.x_grid, terrain_grid, 'k')
ax2.fill_between(pade_vis_4.x_grid, terrain_grid*0, terrain_grid, color='black')
ax1.set_xlabel('Range (km)')
ax1.set_ylabel('Height (m)')
ax2.set_xlabel('Range (km)')
f.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.966, left=0.103, right=0.910, wspace=0.1)
plt.show()

env.z_max = 300
env.M_profile = lambda x, z: profile1d(z)
pade_task_4_elev = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=150e3, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=5,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=4,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      dz_wl=0.8,
                                                      storage=PickleStorage()
                                                  ))
pade_field_4_elev = pade_task_4_elev.calculate()

petool_task_elev = PETOOLPropagationTask(antenna=ant, env=env, two_way=False, max_range_m=150e3, dx_wl=500, n_dx_out=1, dz_wl=3)
petool_field_elev = petool_task_elev.calculate()

env.z_max = 3000
petool_task_m_elev = PETOOLPropagationTask(antenna=ant, env=env, two_way=False, max_range_m=150e3, dx_wl=500, n_dx_out=1,
                                      dz_wl=3, n_dz_out=2)
petool_field_m_elev = petool_task_m_elev.calculate()

pade_vis_4_elev = FieldVisualiser(pade_field_4_elev, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11] (Proposed)', x_mult=1E-3)
petool_vis_elev = FieldVisualiser(petool_field_elev, env=env, trans_func=lambda x: x, label='SSF', x_mult=1E-3)
petool_vis_m_elev = FieldVisualiser(petool_field_m_elev, env=env, trans_func=lambda x: x, label='SSF, z_max=3000 m', x_mult=1E-3)


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 3.2))
z0 = 10
field = np.zeros(len(petool_vis.x_grid))
for i in range(0, len(petool_vis.x_grid)):
    field[i] = petool_vis.field[i, abs(petool_vis.z_grid - petool_vis.env.terrain.elevation(petool_vis.x_grid[i] / petool_vis.x_mult) - z0).argmin()]
ax1.plot(petool_vis.x_grid, field, label=petool_vis.label, color='red')
field = np.zeros(len(petool_vis_m.x_grid))
for i in range(0, len(petool_vis_m.x_grid)):
    field[i] = petool_vis_m.field[i, abs(petool_vis_m.z_grid - petool_vis_m.env.terrain.elevation(petool_vis_m.x_grid[i] / petool_vis_m.x_mult) - z0).argmin()]
ax1.plot(petool_vis_m.x_grid, field, label=petool_vis_m.label, color='blue')
field = np.zeros(len(pade_vis_4.x_grid))
for i in range(0, len(pade_vis_4.x_grid)):
    field[i] = pade_vis_4.field[i, abs(pade_vis_4.z_grid - pade_vis_4.env.terrain.elevation(pade_vis_4.x_grid[i] / pade_vis_4.x_mult) - z0).argmin()]
ax1.plot(pade_vis_4.x_grid, field, label=pade_vis_4.label, color='green')
#ax1.legend()
ax1.set_xlim([0.5, pade_vis_4.x_grid[-1]])
ax1.set_ylim([-100, -5])
ax1.grid()

field = np.zeros(len(petool_vis_elev.x_grid))
for i in range(0, len(petool_vis_elev.x_grid)):
    field[i] = petool_vis_elev.field[i, abs(petool_vis_elev.z_grid - petool_vis_elev.env.terrain.elevation(petool_vis_elev.x_grid[i] / petool_vis_elev.x_mult) - z0).argmin()]
ax2.plot(petool_vis_elev.x_grid, field, label=petool_vis_elev.label, color='red')
field = np.zeros(len(petool_vis_m_elev.x_grid))
for i in range(0, len(petool_vis_m_elev.x_grid)):
    field[i] = petool_vis_m_elev.field[i, abs(petool_vis_m_elev.z_grid - petool_vis_m_elev.env.terrain.elevation(petool_vis_m_elev.x_grid[i] / petool_vis_m_elev.x_mult) - z0).argmin()]
ax2.plot(petool_vis_m_elev.x_grid, field, label=petool_vis_m_elev.label, color='blue')
field = np.zeros(len(pade_vis_4_elev.x_grid))
for i in range(0, len(pade_vis_4_elev.x_grid)):
    field[i] = pade_vis_4_elev.field[i, abs(pade_vis_4_elev.z_grid - pade_vis_4_elev.env.terrain.elevation(pade_vis_4_elev.x_grid[i] / pade_vis_4_elev.x_mult) - z0).argmin()]
ax2.plot(pade_vis_4_elev.x_grid, field, label=pade_vis_4_elev.label, color='green')
ax2.legend()
ax2.set_xlim([0.5, pade_vis_4_elev.x_grid[-1]])
ax2.set_ylim([-100, -5])
ax2.grid()

ax1.set_xlabel('Range (km)')
ax1.set_ylabel('10log|u| (dB)')
ax2.set_xlabel('Range (km)')
f.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.966, left=0.128, right=0.960, wspace=0.1)
plt.show()