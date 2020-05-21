from rwp.environment import *
from rwp.sspade import *
from rwp.vis import *
from propagators.wavenumber import *


logging.basicConfig(level=logging.DEBUG)
environment = Troposphere(flat=True)
environment.ground_material = SaltWater()
environment.z_max = 100
max_range = 100e3

profile1d = interp1d(x=[0, 5, 70, 100, 300], y=[0, 0, -10, 0, 0], fill_value="extrapolate")
environment.M_profile = lambda x, z: profile1d(z)

antenna = GaussAntenna(freq_hz=10000e6, height=25, beam_width=2, eval_angle=0, polarz='H')

sspe_propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna,
                                                        env=environment,
                                                        max_range_m=max_range
                                                        )
environment.rms_m = 1
sspe_propagator_rms = TroposphericRadioWaveSSPadePropagator(antenna=antenna,
                                                        env=environment,
                                                        max_range_m=max_range
                                                        )

sspe_field = sspe_propagator.calculate()
sspe_field_rms = sspe_propagator_rms.calculate()

sspe_vis = FieldVisualiser(sspe_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade + NLBC')
sspe_vis_rms = FieldVisualiser(sspe_field_rms, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade + NLBC RMS')

plt = sspe_vis.plot2d(min=-30, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = sspe_vis_rms.plot2d(min=-30, max=0)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = sspe_vis.plot_hor(25, sspe_vis_rms)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.grid(True)
plt.show()
