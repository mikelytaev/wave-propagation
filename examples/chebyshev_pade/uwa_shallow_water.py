from uwa.field import *
from uwa.source import *
from uwa.environment import *
from uwa.propagators import *
from uwa.ram import *
from uwa.utils import *
from cheb_pade_coefs import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from uwa.vis import AcousticPressureFieldVisualiser2d


logging.basicConfig(level=logging.DEBUG)


src = GaussSource(freq_hz=50, depth=50, beam_width=20, eval_angle=0)
env = UnderwaterEnvironment()
env.sound_speed_profile_m_s = lambda x, z: z*0 + 1500
env.bottom_profile = Bathymetry(ranges_m=[0, 40000, 80000, 120000, 150000], depths_m=[500, 500, 100, 500, 500])
env.bottom_sound_speed_m_s = 1700
env.bottom_density_g_cm = 1
env.bottom_attenuation_dm_lambda = 0.5

max_range = 150000

wavelength = 1500 / src.freq_hz
dz_wl = 1 / wavelength

coefs = cheb_pade_coefs(6, (2, 3), 0.23, 'ratinterp')
cheb_pade_propagator = UnderwaterAcousticsSSPadePropagator(src=src,
                                                           env=env,
                                                           max_range_m=max_range,
                                                           comp_params=HelmholtzPropagatorComputationalParams(z_order=4,
                                                          dx_wl=6,
                                                          dz_wl=dz_wl,
                                                          modify_grid=False,
                                                          exp_pade_coefs=coefs
                                                          ))
cheb_pade_field = cheb_pade_propagator.calculate()
cheb_pade_field.field *= 5.50 #normalization

pade_propagator_f = UnderwaterAcousticsSSPadePropagator(src=src,
                                                           env=env,
                                                           max_range_m=max_range,
                                                           comp_params=HelmholtzPropagatorComputationalParams(z_order=4,
                                                          dx_wl=6,
                                                          dz_wl=dz_wl,
                                                          modify_grid=False,
                                                          exp_pade_order=(2, 3),
                                                          #sqrt_alpha=0,
                                                          ))
pade_field_f = pade_propagator_f.calculate()
pade_field_f.field *= 5.50 #normalization

pade_propagator = UnderwaterAcousticsSSPadePropagator(src=src,
                                                           env=env,
                                                           max_range_m=max_range,
                                                           comp_params=HelmholtzPropagatorComputationalParams(z_order=4,
                                                          dx_wl=6,
                                                          dz_wl=dz_wl,
                                                          modify_grid=False,
                                                          exp_pade_order=(5, 6),
                                                          #sqrt_alpha=0,
                                                          ))
pade_field = pade_propagator.calculate()
pade_field.field *= 5.50 #normalization

cheb_pade_vis = AcousticPressureFieldVisualiser2d(field=cheb_pade_field, label='Rational interpolation-[2/3]')
#cheb_pade_vis.plot2d(-70, 0).show()

pade_vis_f = AcousticPressureFieldVisualiser2d(field=pade_field_f, label='Pade-[2/3]')
#pade_vis_f.plot2d(-70, 0).show()

pade_vis = AcousticPressureFieldVisualiser2d(field=pade_field, label='Pade-[5/6]')
#pade_vis.plot2d(-70, 0).show()

cheb_pade_vis.plot_hor(src.depth, pade_vis_f, pade_vis).show()



f, ax = plt.subplots(1, 3, sharey=True, figsize=(6.4, 3.2), constrained_layout=True)
norm = Normalize(-70, 0)
extent = [cheb_pade_vis.field.x_grid[0]*1e-3, cheb_pade_vis.field.x_grid[-1]*1e-3, cheb_pade_vis.field.z_grid[-1], cheb_pade_vis.field.z_grid[0]]
im = ax[0].imshow(cheb_pade_vis.trans_func(cheb_pade_vis.field.field).real.T[:, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[0].grid()
ax[0].set_ylim([600, 0])
ax[0].set_title(cheb_pade_vis.label)
ax[0].set_ylabel('Depth (m)')
ax[0].set_xlabel('Range (km)')

ax[1].imshow(pade_vis_f.trans_func(pade_vis_f.field.field).real.T[:, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[1].grid()
ax[1].set_ylim([600, 0])
ax[1].set_title(pade_vis_f.label)
ax[1].set_xlabel('Range (km)')

ax[2].imshow(pade_vis.trans_func(pade_vis.field.field).real.T[:, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[2].grid()
ax[2].set_ylim([600, 0])
ax[2].set_title(pade_vis.label)
ax[2].set_xlabel('Range (km)')

f.colorbar(im, ax=ax[:], shrink=0.6, location='bottom')
plt.show()
