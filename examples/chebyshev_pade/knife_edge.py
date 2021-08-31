from rwp.sspade import *
from rwp.vis import *
from scipy.io import loadmat
from cheb_pade_coefs import *


logging.basicConfig(level=logging.DEBUG)

dx_wl = 2
coefs = cheb_pade_coefs(dx_wl, (7, 8), fm.sin(85*fm.pi/180)**2, 'ratinterp')

env = Troposphere(flat=True)
env.z_max = 300
max_propagation_angle = 20
env.knife_edges = [KnifeEdge(range=0.5e3, height=150)]

ant = GaussAntenna(freq_hz=3000e6, height=150, beam_width=4, eval_angle=0, polarz='H')

max_range_m = 1e3

pade_task_f = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=4,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=4,
                                                      dz_wl=0.25,
                                                      z_output_filter=8
                                                  ))
pade_field_f = pade_task_f.calculate()


pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=4,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=4,
                                                      dz_wl=0.25,
                                                      z_output_filter=8
                                                  ))
pade_field = pade_task.calculate()

cheb_pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=4,
                                                      exp_pade_coefs=coefs,
                                                      dx_wl=dx_wl,
                                                      x_output_filter=4,
                                                      dz_wl=0.25,
                                                      z_output_filter=8
                                                  ))
cheb_pade_field = cheb_pade_task.calculate()

pade_vis_f = FieldVisualiser(pade_field_f, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[4/5]', x_mult=1E-3)

pade_vis = FieldVisualiser(pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11]', x_mult=1E-3)

cheb_pade_vis = FieldVisualiser(cheb_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)),
                                label='Rational interpolation-[4/5]', x_mult=1E-3)

plt = cheb_pade_vis.plot_hor_over_terrain(5, pade_vis_f, pade_vis)
plt.xlabel('Range (km)')
plt.ylabel('20log|u| (dB)')
plt.xlim([2, 8])
plt.ylim([-120, -20])
plt.grid(True)
plt.tight_layout()
plt.show()

f, ax = plt.subplots(1, 3, sharey=True, figsize=(6, 3.2), constrained_layout=True)
norm = Normalize(-120, -20)
extent = [pade_vis.x_grid[0], pade_vis.x_grid[-1], pade_vis.z_grid[0], pade_vis.z_grid[-1]]

im = ax[0].imshow(pade_vis_f.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
terrain_grid = np.array([pade_vis_f.env.terrain.elevation(v) for v in pade_vis_f.x_grid / pade_vis_f.x_mult])
ax[0].plot(pade_vis_f.x_grid, terrain_grid, 'k')
ax[0].fill_between(pade_vis_f.x_grid, terrain_grid*0, terrain_grid, color='brown')
ax[0].grid()
ax[0].set_title(pade_vis_f.label)
ax[0].set_xlabel('Range (km)')
ax[0].set_ylabel('Height (m)')
#ax[0].set_xlim([2.5, 4.5])
ax[0].set_ylim([0, 200])

ax[1].imshow(pade_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
terrain_grid = np.array([pade_vis.env.terrain.elevation(v) for v in pade_vis.x_grid / pade_vis.x_mult])
ax[1].plot(pade_vis.x_grid, terrain_grid, 'k')
ax[1].fill_between(pade_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')
ax[1].grid()
ax[1].set_title(pade_vis.label)
ax[1].set_xlabel('Range (km)')
#ax[1].set_xlim([2.5, 4.5])
ax[1].set_ylim([0, 200])

ax[2].imshow(cheb_pade_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
terrain_grid = np.array([cheb_pade_vis.env.terrain.elevation(v) for v in cheb_pade_vis.x_grid / cheb_pade_vis.x_mult])
ax[2].plot(cheb_pade_vis.x_grid, terrain_grid, 'k')
ax[2].fill_between(cheb_pade_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')
ax[2].grid()
ax[2].set_title('Rat. interp.- [4/5]')
ax[2].set_xlabel('Range (km)')
#ax[2].set_xlim([2.5, 4.5])
ax[2].set_ylim([0, 200])

f.colorbar(im, ax=ax[:], shrink=0.6, location='bottom')
plt.show()