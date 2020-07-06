from rwp.environment import *
from rwp.sspade import *
from rwp.vis import *
from rwp.petool import *

logging.basicConfig(level=logging.DEBUG)
environment = Troposphere(flat=True)
environment.ground_material = SaltWater()
#environment.rms_m = 1
environment.z_max = 300
max_range = 1000

antenna = GaussAntenna(freq_hz=10e9, height=30, beam_width=2, eval_angle=0, polarz='V')

propagator = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range,
                                                   comp_params=HelmholtzPropagatorComputationalParams(
                                                       max_propagation_angle=3,
                                                       modify_grid=False,
                                                       z_order=4,
                                                       exp_pade_order=(7, 8),
                                                       dx_wl=1000,
                                                       inv_z_transform_tau=1.55
                                                   ))

field = propagator.calculate()

vis = FieldVisualiser(field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade + Transparent BC', x_mult=1E-3)

# plt = vis.plot_hor(30)
# plt.xlabel('Range (km)')
# plt.ylabel('10log|u| (dB)')
# plt.tight_layout()
# plt.show()
#
# plt = vis.plot2d(min=-70, max=0)
# plt.title('10log|u|')
# plt.xlabel('Range (km)')
# plt.ylabel('Height (m)')
# plt.tight_layout()
# plt.show()

coefs = propagator.propagator.lower_bc.coefs
abs_coefs = np.array([np.linalg.norm(a) for a in coefs])
plt.plot(np.log10(abs_coefs))
plt.show()