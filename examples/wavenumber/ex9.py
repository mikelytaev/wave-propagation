from rwp.sspade import *
from rwp.vis import *
from rwp.environment import *
from propagators.wavenumber import *

logging.basicConfig(level=logging.DEBUG)
environment = Troposphere(flat=True)
environment.z_max = 100
environment.ground_material = PerfectlyElectricConducting()

freq_hz = 3000e6

antenna = GaussAntenna(freq_hz=freq_hz, height=50, beam_width=2, eval_angle=0, polarz='V')
max_range = 10000

pade_params = HelmholtzPropagatorComputationalParams(two_way=False,
                                                     exp_pade_order=(7, 8),
                                                     max_propagation_angle=abs(antenna.eval_angle)+3,
                                                     z_order=4,
                                                     dx_wl=500,
                                                     inv_z_transform_tau=1/1.01,
                                                     #tol=1e-3
                                                     )
pade_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range, comp_params=pade_params)
pade_field = pade_task.calculate()

pade_vis = FieldVisualiser(pade_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade + NLBC', x_mult=1)

plt = pade_vis.plot2d(min=-70, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.show()

plt = pade_vis.plot_hor(50)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.grid(True)
plt.show()