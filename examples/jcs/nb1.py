from rwp.sspade import *
from rwp.vis import *
from scipy.io import loadmat
from examples.chebyshev_pade.cheb_pade_coefs import *


logging.basicConfig(level=logging.DEBUG)

max_propagation_angle = 20
dx_wl = 50
wl = 0.1
max_range_m = 10e3
coefs, a0 = cheb_pade_coefs(dx_wl, (6, 7), fm.sin(max_propagation_angle*fm.pi/180)**2, 'ratinterp')

env = Troposphere(flat=True)
env.z_max = 300
env.terrain = Terrain(elevation=lambda x: pyramid2(x, 20, 150, 5e3), ground_material=PerfectlyElectricConducting())
#env.knife_edges = [KnifeEdge(range=3e3, height=150)]

ant = GaussAntenna(wavelength=wl, height=150, beam_width=4, eval_angle=0, polarz='H')


cheb_pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=4,
                                                      exp_pade_coefs=coefs,
                                                      exp_pade_a0_coef=a0,
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=0.25,
                                                      z_output_filter=8,
                                                      two_way=False
                                                  ))
cheb_pade_field = cheb_pade_task.calculate()

from rwp.petool import PETOOLPropagationTask
petool_task_elevated = PETOOLPropagationTask(antenna=ant,
                                             env=env,
                                             two_way=False,
                                             max_range_m=max_range_m,
                                             dx_wl=dx_wl,
                                             n_dx_out=1,
                                             dz_wl=0.25,
                                             n_dz_out=8)
petool_field_elevated = petool_task_elevated.calculate()


cheb_pade_vis = FieldVisualiser(cheb_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)),
                                label='AAA-[7/7]', x_mult=1E-3)


import matplotlib as mpl
mpl.rcParams['axes.titlesize'] = 'medium'
f, ax = plt.subplots(1, 1, sharey=True, figsize=(6, 3.2), constrained_layout=True)
norm = Normalize(0, 10)
extent = [cheb_pade_field.x_grid[0]*1e-3, cheb_pade_field.x_grid[-1]*1e-3, cheb_pade_field.z_grid[0], cheb_pade_field.z_grid[-1]]

err = np.abs(2*petool_field_elevated.field - 20*np.log10(np.abs(cheb_pade_field.field[1:,:])+1e-16))
im = ax.imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax.grid()
ax.set_title('Δz=2.0λ, 2nd order')
ax.set_ylabel('Height (m)')

terrain_grid = np.array([cheb_pade_vis.env.terrain.elevation(v) for v in cheb_pade_vis.x_grid / cheb_pade_vis.x_mult])
ax.plot(cheb_pade_vis.x_grid, terrain_grid, 'k')
ax.fill_between(cheb_pade_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')

f.colorbar(im, shrink=0.6, location='bottom')
#f.tight_layout()
plt.show()

petool_vis = FieldVisualiser(petool_field_elevated, env=env, trans_func=lambda x: 2 * x, label='Метод расщ. Фурье', x_mult=1E-3)
plt = petool_vis.plot2d(0, -100, True)
plt.show()

plt = cheb_pade_vis.plot_hor_over_terrain(20, petool_vis)
plt.xlabel('Range (km)')
plt.ylabel('20log|u| (dB)')
#plt.xlim([2, 8])
plt.ylim([-120, -20])
plt.grid(True)
plt.tight_layout()
plt.show()