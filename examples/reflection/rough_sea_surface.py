from rwp.environment import *
from rwp.sspade import *
from rwp.vis import *


logging.basicConfig(level=logging.DEBUG)
environment = Troposphere()
environment.ground_material = SaltWater()
environment.z_max = 300
max_range = 200e3

profile1d = interp1d(x=[0, 50, 1050], y=[330-330, 320-330, 438-330], fill_value="extrapolate")
environment.M_profile = lambda x, z: profile1d(z)

antenna = GaussAntenna(wavelength=0.03, height=25, beam_width=2, eval_angle=0, polarz='H')

#propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range)
#field = propagator.calculate()

environment.rms_m = 1
propagator_rms = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range,
                                                       comp_params=HelmholtzPropagatorComputationalParams(
                                                           exp_pade_order=(5, 5),
                                                           sqrt_alpha=0,
                                                           dx_wl=900,
                                                           z_order=4,
                                                           max_propagation_angle=1,
                                                           dz_wl=1,
                                                           tol=1e-7
                                                       ))
field_rms = propagator_rms.calculate()

#vis = FieldVisualiser(field.path_loss(), label='Pade + NLBC', x_mult=1E-3)
vis_rms = FieldVisualiser(field_rms.path_loss(), label='Pade + NLBC RMS=1', x_mult=1E-3)

plt = vis_rms.plot_hor(25)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.grid(True)
plt.gca().invert_yaxis()
plt.show()

# plt = vis.plot2d(min=50, max=170)
# plt.title('10log|u|')
# plt.xlabel('Range (km)')
# plt.ylabel('Height (m)')
# plt.tight_layout()
# plt.show()

plt = vis_rms.plot2d(min=50, max=170)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()
